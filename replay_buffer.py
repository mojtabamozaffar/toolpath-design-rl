import numpy as np
import networks

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

    def get_batch(self):
        observation_batch, action_batch, reward_batch, value_batch, policy_batch, gradient_scale_batch = ([], [], [], [], [], [])

        for _ in range(self.config.batch_size):
            game_history = self.sample_game(self.buffer)
            game_pos = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(game_history, game_pos)

            observation_batch.append(game_history.get_stacked_observations(game_pos, self.config.stacked_observations))
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append([min(self.config.num_unroll_steps, len(game_history.action_history) - game_pos)]* len(actions))

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return observation_batch, action_batch, value_batch, reward_batch, policy_batch, gradient_scale_batch

    def sample_game(self, buffer):
        return np.random.choice(self.buffer)

    def sample_position(self, game_history):
        return np.random.choice(range(len(game_history.reward_history)))
    
    def compute_target_value(self, game_history, index):
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            if self.config.use_last_model_value:
                observation = (torch.tensor(game_history.get_stacked_observations(bootstrap_index, self.config.stacked_observations))
                    .float().to(torch.device(self.config.device))
                    .unsqueeze(0))
                last_step_value = networks.support_to_scalar(self.model.initial_inference(observation)[0], self.config.support_size_value).item()
            else:
                last_step_value = game_history.root_values[bootstrap_index]

            value = last_step_value * self.config.discount ** self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(game_history.reward_history[index + 1 : bootstrap_index + 1]):
            value += reward * self.config.discount ** i
            
        return value

    def make_target(self, game_history, state_index):
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append([1 / len(game_history.child_visits[0])for _ in range(len(game_history.child_visits[0]))])
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append([1 / len(game_history.child_visits[0])for _ in range(len(game_history.child_visits[0]))])
                actions.append(np.random.choice(game_history.action_history))

        return target_values, target_rewards, target_policies, actions