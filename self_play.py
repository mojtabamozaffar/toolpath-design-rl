import numpy as np
import torch
import math
import ray
import copy
import networks


@ray.remote        
def play_one_game(model, env_func, config, temperature, save=False, filename = ''):
    game_history = GameHistory()
    game = env_func(max_steps = config.max_moves, window_size = config.observation_shape[1])
    observation = game.reset()
    game_history.action_history.append(0)
    game_history.observation_history.append(observation)
    game_history.reward_history.append(0)
    done = False

    with torch.no_grad():
        while (not done and len(game_history.action_history) <= config.max_moves):
            root = MCTS(config).run(model, observation, game.actions,
                False if temperature == 0 else True)

            action = select_action(root, temperature 
                    if len(game_history.action_history) < config.temperature_threshold else 0)
            observation, reward, done, _ = game.step(action)

            game_history.store_search_statistics(root, [i for i in range(config.action_space_size)])
            game_history.action_history.append(action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
    if save:
        game.plot_toolpath(save = True, folder = config.logdir, filename = filename)

    game.close()
    return game_history


def select_action(node, temperature):
    visit_counts = np.array(
        [child.visit_count for child in node.children.values()]
    )
    actions = [action for action in node.children.keys()]
    if temperature == 0:
        action = actions[np.argmax(visit_counts)]
    elif temperature == float("inf"):
        action = np.random.choice(actions)
    else:
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(
            visit_count_distribution
        )
        action = np.random.choice(actions, p=visit_count_distribution)

    return action


class MCTS:
    def __init__(self, config):
        self.config = config

    def run(self, model, observation, legal_actions, add_exploration_noise):
        root = Node(0)
        observation = (torch.tensor(observation).float().unsqueeze(0).to(next(model.parameters()).device))
        _, reward, policy_logits, hidden_state = model.initial_inference(observation)
        reward = networks.support_to_scalar(reward, self.config.support_size_reward).item()
        root.expand(legal_actions, reward, policy_logits, hidden_state)
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = networks.support_to_scalar(value, self.config.support_size_value).item()
            reward = networks.support_to_scalar(reward, self.config.support_size_reward).item()
            node.expand(
                [i for i in range(self.config.action_space_size)],
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(
                search_path, value, min_max_stats
            )

        return root

    def select_child(self, node, min_max_stats):
        max_ucb = max(self.ucb_score(node, child, min_max_stats) for action, child in node.children.items())
        action = np.random.choice([action for action, child in node.children.items() if self.ucb_score(node, child, min_max_stats) == max_ucb])
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            value_score = min_max_stats.normalize(
                child.reward + self.config.discount * child.value()
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        for node in reversed(search_path):
            node.value_sum += value #if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * node.value())
            value = node.reward + self.config.discount * value


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, reward, policy_logits, hidden_state):
        self.reward = reward
        self.hidden_state = hidden_state
        policy = {}
        for a in actions:
            try:
                policy[a] = 1/sum(torch.exp(policy_logits[0] - policy_logits[0][a]))
            except OverflowError:
                print("Warning: prior has been approximated")
                policy[a] = 0.0
        for action, p in policy.items():
            self.children[action] = Node(p)
            
    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

class GameHistory:
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.child_visits = []
        self.root_values = []

    def store_search_statistics(self, root, action_space):
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append([root.children[a].visit_count / sum_visits
                                      if a in root.children else 0 for a in action_space])
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)


class MinMaxStats:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
