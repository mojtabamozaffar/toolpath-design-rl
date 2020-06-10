import numpy
import torch
import math
from graphviz import Graph
import matplotlib.pyplot as plt
import ray

# class SelfPlay:
#     def __init__(self, shared_storage, replay_buffer, game, config, test_mode=False):
#         self.config = config
#         self.game = game
#         self.shared_storage = shared_storage
#         self.replay_buffer = replay_buffer
#         self.test_mode = test_mode

#     def play(self, it):
#         total_rewards = []
#         model = self.shared_storage.current_network
#         model.eval()
#         for i in range(self.config.n_episodes if not self.test_mode else self.config.eval_episodes):
#             game_history = self.play_one_game(model)
#             if self.test_mode:
#                 total_rewards.append(sum(game_history.reward_history))
#                 fig = self.game.env.plot_toolpath()
#                 fig.savefig(self.config.logdir + '/' + 'toolpath_' + str(it)+'_test_'+str(i)+'.png', dpi=300)
#                 plt.close()
#             else:
#                 self.replay_buffer.save_game(game_history)
#         if self.test_mode:
#             self.shared_storage.set_infos("total_reward", sum(total_rewards)/self.config.eval_episodes)
#         else:
#             fig = self.game.env.plot_toolpath()
#             fig.savefig(self.config.logdir + '/' + 'toolpath_' + str(it)+'_train'+'.png', dpi=300)
#             plt.close()

@ray.remote
def play_one_game(model, env_func, config, temperature, section_id=None, loop=0, save=False, self_play_type = 'test'):
        game_history = GameHistory()
        #observation = self.game.reset()
        # observation = self.stack_previous_observations(observation, game_history, self.config.stacked_observations)
        game = env_func(max_steps = config.max_moves, section_id=section_id)
        observation = game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(game.to_play())

        done = False
        
        #temperature = (0 if self.test_mode else 
        #               self.config.visit_softmax_temperature_fn(self.shared_storage.get_infos()["training_step"]))

        with torch.no_grad():
            while (not done and len(game_history.action_history) <= config.max_moves):
                root = MCTS(config).run(model, observation, game.actions, 
                    game.to_play(),
                    False if temperature == 0 else True)

                #action = select_action(root, temperature)
                
                action = select_action(root, temperature 
                        if len(game_history.action_history) < config.temperature_threshold else 0)
                observation, reward, done, _ = game.step(action)

                # observation = self.stack_previous_observations(observation, game_history, self.config.stacked_observations,)

                game_history.store_search_statistics(root, [i for i in range(config.action_space_size)])

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(game.to_play())

        game.close()
        return game_history

def stack_previous_observations(observation, game_history, num_stacked_observations):
    stacked_observations = observation.copy()
    for i in range(num_stacked_observations):
        try:
            previous_observation = numpy.concatenate(
                (
                    game_history.observation_history[-i - 1][
                        : observation.shape[0]
                    ],
                    [numpy.ones_like(observation[0])
                    * game_history.action_history[-i - 1]],
                ), axis=0
            )
        except IndexError:
            previous_observation = numpy.concatenate(
                (numpy.zeros_like(observation), [numpy.zeros_like(observation[0])]), axis=0
            )
        stacked_observations = numpy.concatenate(
            (stacked_observations, previous_observation), axis=0
        )
    return stacked_observations


def select_action(node, temperature):
    """
    Select action according to the visit count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function 
    in the config.
    """
    visit_counts = numpy.array(
        [child.visit_count for child in node.children.values()]
    )
    actions = [action for action in node.children.keys()]
    if temperature == 0:
        action = actions[numpy.argmax(visit_counts)]
    elif temperature == float("inf"):
        action = numpy.random.choice(actions)
    else:
        # See paper appendix Data Generation
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(
            visit_count_distribution
        )
        action = numpy.random.choice(actions, p=visit_count_distribution)

    return action


# Game independant
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(self, model, observation, legal_actions, to_play, add_exploration_noise):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        root = Node(0)
        observation = (torch.tensor(observation).float().unsqueeze(0).to(self.config.device))
        _, reward, policy_logits, hidden_state = model.initial_inference(observation)
        reward = self.support_to_scalar(reward, self.config.support_size_reward).item()
        root.expand(
            legal_actions, to_play, reward, policy_logits, hidden_state,
        )
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(self.config.device),
            )
            value = self.support_to_scalar(value, self.config.support_size_value).item()
            reward = self.support_to_scalar(reward, self.config.support_size_reward).item()
            node.expand(
                [i for i in range(self.config.action_space_size)],
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(
                search_path, value, virtual_to_play, min_max_stats
            )

        return root

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        _, action, child = max(
            (self.ucb_score(node, child, min_max_stats), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
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

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.config.discount * value

    @staticmethod
    def support_to_scalar(logits, support_size):
        """
        Transform a categorical representation to a scalar
        See paper appendix Network Architecture
        """
        # Decode to a scalar
        probabilities = torch.softmax(logits, dim=1)
        support = (
            torch.tensor([x for x in range(-support_size, support_size + 1)])
            .expand(probabilities.shape)
            .float()
            .to(device=probabilities.device)
        )
        x = torch.sum(support * probabilities, dim=1, keepdim=True)

        # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
        x = torch.sign(x) * (
            ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
            ** 2
            - 1
        )
        return x


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
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

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
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
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
            
    def print_stat(self):
        print("Node stats")
        print("visit count: {}".format(self.visit_count))
        print("value: {:.2f}".format(self.value() if self.value() else 0.0))
        print("reward: {:.2f}".format(self.reward))
        if self.expanded():
            child_visits = [child.visit_count for child in self.children.values()]
            print("child visits: {}, {}, {}, {}, {}, {}, {}, {}".format(child_visits[0], child_visits[1], child_visits[2], child_visits[3], child_visits[4], child_visits[5], child_visits[6], child_visits[7]))
            child_prior = [child.prior for child in self.children.values()]
            print("child prior: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(child_prior[0], child_prior[1], child_prior[2], child_prior[3], child_prior[4], child_prior[5], child_prior[6], child_prior[7]))
        else:
            print("unexpanded")
        print("")
        
    def plot_tree(self, max_depth = 3):
        g = Graph('G', filename = 'graph', format='eps')
        g.node('0', "visited: {:2}\nvalue: {:.2f}\nreward: {:.2f}".format(self.visit_count, self.value(), self.reward))
        if self.expanded():
            expand_graph(g, self, '0', depth=0, max_depth = max_depth)
        g.render()
                
def expand_graph(g, node, base_id, depth, max_depth):
    if not node.expanded() or depth > max_depth:
        return
    for ind, child in enumerate(node.children.values()):
        if child.visit_count > 0:
            g.node(base_id+'.'+str(ind), "visited: {:2}\nvalue: {:.2f}\nreward: {:.2f}".format(child.visit_count, child.value(), child.reward))
        else:
            g.node(base_id+'.'+str(ind), "visited: 0\nvalue: 0.00\nreward: 0.00")
        g.edge(base_id, base_id+'.'+str(ind), label='prior: {:.2f}'.format(child.prior))
        expand_graph(g, child, base_id+'.'+str(ind), depth+1, max_depth)


class GameHistory:
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []

    def store_search_statistics(self, root, action_space):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(
            [root.children[a].visit_count / sum_visits if a in root.children else 0 for a in action_space])
        self.root_values.append(root.value())


class MinMaxStats:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
