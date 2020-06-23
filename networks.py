import torch

def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class RepresentationNetwork(torch.nn.Module):
    def __init__(self, observation_shape, stacked_observations, num_blocks, num_channels):
        super(RepresentationNetwork, self).__init__()
        self.conv = conv3x3(observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels, stride=2)
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(num_channels, num_channels, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)
        self.relu2 = torch.nn.ReLU()
        # self.conv3 = conv3x3(num_channels, num_channels, stride=2)
        # self.bn3 = torch.nn.BatchNorm2d(num_channels)
        # self.relu3 = torch.nn.ReLU()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu3(out)
        for block in self.resblocks:
            out = block(out)
        return out


class DynamicNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        reduced_channels,
        fc_reward_layers,
        full_support_size,
    ):
        super(DynamicNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.relu = torch.nn.ReLU()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        self.conv1x1 = torch.nn.Conv2d(num_channels - 1, reduced_channels, 1)
        self.block_output_size = (
            reduced_channels * observation_shape[1] * observation_shape[2]
        )
        self.fc = FullyConnectedNetwork(
            self.block_output_size,
            fc_reward_layers,
            full_support_size,
            activation=None,
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for block in self.resblocks:
            out = block(out)
        state = out
        out = self.conv1x1(out)
        out = out.view(-1, self.block_output_size)
        reward = self.fc(out)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
    ):
        super(PredictionNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1 = torch.nn.Conv2d(num_channels, reduced_channels, 1)
        self.block_output_size = (
            reduced_channels * observation_shape[1] * observation_shape[2]
        )
        self.fc_value = FullyConnectedNetwork(
            self.block_output_size, fc_value_layers, full_support_size, activation=None,
        )
        self.fc_policy = FullyConnectedNetwork(
            self.block_output_size,
            fc_policy_layers,
            action_space_size,
            activation=None,
        )

    def forward(self, x):
        out = x
        for block in self.resblocks:
            out = block(out)
        out = self.conv1x1(out)
        out = out.view(-1, self.block_output_size)
        value = self.fc_value(out)
        policy = self.fc_policy(out)
        return policy, value

class MuZeroResidualNetwork(torch.nn.Module):
    def __init__(self, config):
        observation_shape = config.observation_shape
        stacked_observations = 0
        action_space_size = config.action_space_size
        num_blocks = config.blocks
        num_channels = config.channels
        reduced_channels = config.reduced_channels
        fc_reward_layers = config.resnet_fc_reward_layers
        fc_value_layers = config.resnet_fc_value_layers
        fc_policy_layers = config.resnet_fc_policy_layers
        support_size_value = config.support_size_value
        support_size_reward = config.support_size_reward
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size_value = 2 * support_size_value + 1
        self.full_support_size_reward = 2 * support_size_reward + 1

        self.representation_network = RepresentationNetwork(
            observation_shape, stacked_observations, num_blocks, num_channels
        )

        self.dynamics_network = DynamicNetwork(
            (8, 8, 8),
            num_blocks,
            num_channels + 1,
            reduced_channels,
            fc_reward_layers,
            self.full_support_size_reward,
        )

        self.prediction_network = PredictionNetwork(
            (8, 8, 8),
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels,
            fc_value_layers,
            fc_policy_layers,
            self.full_support_size_value,
        )

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        encoded_state_normalized = self.normalize_encoded_state(encoded_state)
        return encoded_state_normalized

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def dynamics(self, encoded_state, action):
        encoded_state_stacked_action = self.stack_action(encoded_state, action)
        next_encoded_state, reward = self.dynamics_network(encoded_state_stacked_action)
        next_encoded_state_normalized = self.normalize_encoded_state(next_encoded_state)
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = (
            torch.zeros(1, self.full_support_size_reward)
            .scatter(1, torch.tensor([[self.full_support_size_reward // 2]]).long(), 1.0)
            .repeat(len(observation), 1)
            .to(observation.device)
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

    def pretrain_inference(self, observation, action):
        encoded_state = self.representation(observation)
        _, value = self.prediction(encoded_state)

        _, reward = self.dynamics(encoded_state, action)

        return (
            value,
            reward
        ) 

    def get_weights(self):
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def normalize_encoded_state(self, encoded_state):
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state == 0] = 1
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def stack_action(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None].float() * action_one_hot / self.action_space_size
        )
        encoded_state_stacked_action = torch.cat((encoded_state, action_one_hot), dim=1)
        return encoded_state_stacked_action

class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=None):
        super(FullyConnectedNetwork, self).__init__()
        size_list = [input_size] + layer_sizes
        layers = []
        if 1 < len(size_list):
            for i in range(len(size_list) - 1):
                layers.extend(
                    [
                        torch.nn.Linear(size_list[i], size_list[i + 1]),
                        torch.nn.LeakyReLU(),
                    ]
                )
        layers.append(torch.nn.Linear(size_list[-1], output_size))
        if activation:
            layers.append(activation)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def support_to_scalar(logits, support_size):
    probabilities = torch.softmax(logits, dim=1)
    support = (torch.tensor([x for x in range(-support_size, support_size + 1)])
               .expand(probabilities.shape)
               .float()
               .to(device=probabilities.device))
    x = torch.sum(support * probabilities, dim=1, keepdim=True)
    x = torch.sign(x) * (((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))** 2 - 1)
    return x


def scalar_to_support(x, support_size):
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1))
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits