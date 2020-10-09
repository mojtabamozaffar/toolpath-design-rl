import numpy as np
import ray
import config as gcofig

class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer = []
        self.self_play_count = 0

    def save_game(self, game_history):
        if len(self.buffer) > self.config.window_size:
            self.buffer.pop(0)
        self.buffer.append(game_history)
        self.self_play_count += 1

    def get_self_play_count(self):
        return self.self_play_count

def get_batch(buffer, config):
    observation_batch, action_batch, reward_batch, value_batch, policy_batch = (
        [],
        [],
        [],
        [],
        [],
    )

    for _ in range(config.batch_size):
        game_history = sample_game(buffer)
        game_pos = sample_position(game_history)

        value, reward, policy, actions = make_target(game_history, game_pos, config)

        observation_batch.append(game_history.observation_history[game_pos])
        action_batch.append(actions)
        value_batch.append(value)
        reward_batch.append(reward)
        policy_batch.append(policy)

    # observation_batch: batch, channels, heigth, width
    # action_batch: batch, num_unroll_steps+1
    # value_batch: batch, num_unroll_steps+1
    # reward_batch: batch, num_unroll_steps+1
    # policy_batch: batch, num_unroll_steps+1, len(action_space)
    
    return observation_batch, action_batch, value_batch, reward_batch, policy_batch

def sample_game(buffer):
    return np.random.choice(buffer)

def sample_position(game_history):
    return np.random.choice(range(len(game_history.reward_history)))

def make_target(game_history, state_index, config):
    target_values, target_rewards, target_policies, actions = [], [], [], []
    for current_index in range(
        state_index, state_index + config.num_unroll_steps + 1
    ):
        bootstrap_index = current_index + config.td_steps
        if bootstrap_index < len(game_history.root_values):
            value = (
                game_history.root_values[bootstrap_index]
                * config.discount ** config.td_steps
            )
        else:
            value = 0
            # n_p = np.sum(game_history.observation_history[-1])
            # value = (1-config.discount**n_p)/(1-config.discount)

        for i, reward in enumerate(
            game_history.reward_history[current_index+1:bootstrap_index+1]
        ):
            value += reward * config.discount ** i

        if current_index < len(game_history.root_values):
            target_values.append(value)
            target_rewards.append(game_history.reward_history[current_index])
            target_policies.append(game_history.child_visits[current_index])
            actions.append(game_history.action_history[current_index])
        elif current_index == len(game_history.root_values):
            target_values.append(0)
            target_rewards.append(game_history.reward_history[current_index])
            target_policies.append(
                [
                    1 / len(game_history.child_visits[0])
                    for _ in range(len(game_history.child_visits[0]))
                ]
            )
            actions.append(game_history.action_history[current_index])
        else:
            # States past the end of games are treated as absorbing states
            target_values.append(0)
            target_rewards.append(0)
            # Uniform policy to give the tensor a valid dimension
            target_policies.append(
                [
                    1 / len(game_history.child_visits[0])
                    for _ in range(len(game_history.child_visits[0]))
                ]
            )
            actions.append(np.random.choice(game_history.action_history))

    return target_values, target_rewards, target_policies, actions

if gcofig.use_ray:
    get_batch = ray.remote(get_batch)