import numpy
import torch

class Trainer:
    def __init__(self, shared_storage, config):
        self.config = config
        self.shared_storage = shared_storage
        #self.replay_buffer = replay_buffer
        self.training_step = 0

        self.model = shared_storage.network
        # self.model.to(torch.device(config.device))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_init,
            weight_decay=self.config.weight_decay)

    def train(self, batches):
        # self.model.train()
        for i in range(self.config.n_epochs):
            # batch = self.replay_buffer.get_batch()
            self.update_lr()
            total_loss, value_loss, reward_loss, policy_loss = self.update_weights(batches[i])
            
            self.shared_storage.set_infos("training_step", self.training_step)
            self.shared_storage.set_infos("lr", self.optimizer.param_groups[0]["lr"])
            self.shared_storage.set_infos("total_loss", total_loss)
            self.shared_storage.set_infos("value_loss", value_loss)
            self.shared_storage.set_infos("reward_loss", reward_loss)
            self.shared_storage.set_infos("policy_loss", policy_loss)


    def update_weights(self, batch):

        (observation_batch, action_batch, target_value, target_reward, target_policy) = batch

        device = next(self.model.parameters()).device
        observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).float().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        # observation_batch: batch, channels, heigth, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)

        target_value = self.scalar_to_support(target_value, self.config.support_size_value)
        target_reward = self.scalar_to_support(target_reward, self.config.support_size_reward)
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        
        # Generate predictions
        value, reward, policy_logits, hidden_state = self.model.initial_inference(observation_batch)
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(hidden_state, action_batch[:, i])
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps + 1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        # Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        # Ignore reward loss for the first batch step
        value, reward, policy_logits = predictions[0]
        (current_value_loss, _, current_policy_loss) = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )
            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

        value_loss *= self.config.value_loss_weight
        loss = (value_loss + reward_loss + policy_loss).mean()

        # Scale gradient by number of unroll steps (See paper Training appendix)
        loss.register_hook(lambda grad: grad / self.config.num_unroll_steps)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (loss.item(),value_loss.mean().item(),reward_loss.mean().item(),policy_loss.mean().item())

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def scalar_to_support(x, support_size):
        """
        Transform a scalar to a categorical representation with (2 * support_size + 1) categories
        See paper appendix Network Architecture
        """
        # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
        x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

        # Encode on a vector
        x = torch.clamp(x, -support_size, support_size)
        floor = x.floor()
        prob = x - floor
        logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
        logits.scatter_(
            2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
        )
        indexes = floor + support_size + 1
        prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
        indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
        logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
        return logits

    @staticmethod
    def loss_function(
        value, reward, policy_logits, target_value, target_reward, target_policy
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1).mean()
        reward_loss = (
            (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1).mean()
        )
        policy_loss = (
            (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(1).mean()
        )
        return value_loss, reward_loss, policy_loss
