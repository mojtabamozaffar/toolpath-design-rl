import torch



def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.relu1 = torch.nn.LeakyReLU()
        self.conv2 = conv3x3(num_channels, num_channels)
        self.relu2 = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out += x
        out = self.relu2(out)
        return out


class RepresentationNetwork(torch.nn.Module):
    def __init__(self, observation_shape, stacked_observations):
        super(RepresentationNetwork, self).__init__()
        self.conv = torch.nn.Conv2d(observation_shape[0] * (stacked_observations + 1) + stacked_observations, 
                                    32, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(32, 
                                    64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu2 = torch.nn.LeakyReLU()
        self.conv3 = torch.nn.Conv2d(64, 
                                    64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        return out


class DynamicNetwork(torch.nn.Module):
    def __init__(self, observation_shape):
        super(DynamicNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.fc_act = torch.nn.Linear(8, 16)
        self.relu_act = torch.nn.LeakyReLU()
        self.conv = conv3x3(65, 64)
        self.relu = torch.nn.LeakyReLU()
        self.resblocks = torch.nn.ModuleList([ResidualBlock(64) for _ in range(4)])
        self.block_output_size = (64 * observation_shape[1] * observation_shape[2])
        self.fc = FullyConnectedNetwork(self.block_output_size, [128, 32], 1)
        self.conv2 = conv3x3(64, 64)
        self.relu2 = torch.nn.LeakyReLU()

    def forward(self, x, action):
        act = self.relu_act(self.fc_act(action)).view(-1, 1, 4, 4)
        out = torch.cat((x, act), dim=1)
        out = self.relu(self.conv(out))      
        for block in self.resblocks:
            out = block(out) 
        reward = out.view(-1, self.block_output_size)
        reward = self.fc(reward)
        state = self.relu2(self.conv2(out))  
        return state, reward

class PredictionNetwork(torch.nn.Module):
    def __init__(self,observation_shape, action_space_size):
        super(PredictionNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.resblocks = torch.nn.ModuleList([ResidualBlock(64) for _ in range(4)])
        self.conv1x1 = torch.nn.Conv2d(64, 64, 1)
        self.relu = torch.nn.LeakyReLU()
        self.block_output_size = (64 * observation_shape[1] * observation_shape[2])
        self.fc_value = FullyConnectedNetwork(self.block_output_size, [128, 32], 1)
        self.fc_policy = FullyConnectedNetwork(self.block_output_size, [128, 32], action_space_size)

    def forward(self, x):
        out = x
        for block in self.resblocks:
            out = block(out)
        out = self.relu(self.conv1x1(out))
        out = out.view(-1, self.block_output_size)
        value = self.fc_value(out)
        policy = self.fc_policy(out)
        return policy, value

class MuZeroResidualNetwork(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        observation_shape = config.observation_shape
        stacked_observations = 0
        self.action_space_size = config.action_space_size
        
        self.representation_network = RepresentationNetwork(observation_shape, stacked_observations)

        self.dynamics_network = DynamicNetwork((64, 4, 4))

        self.prediction_network = PredictionNetwork((64, 4, 4), self.action_space_size)

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        encoded_state_normalized = self.normalize_encoded_state(encoded_state)
        return encoded_state_normalized

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def dynamics(self, encoded_state, action):
        one_hot_action = self.stack_action(action)
        next_encoded_state, reward = self.dynamics_network(encoded_state, one_hot_action)
        next_encoded_state_normalized = self.normalize_encoded_state(next_encoded_state)
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # TODO: check
        # reward equal to 0 for consistency
        # reward = (
        #     torch.zeros(1, self.full_support_size_reward)
        #     .scatter(1, torch.tensor([[self.full_support_size_reward // 2]]).long(), 1.0)
        #     .repeat(len(observation), 1)
        #     .to(observation.device)
        # )

        # reward = (
        #     torch.zeros(1, self.full_support_size_reward)
        #     .scatter(1, torch.tensor([[0]]).long(), 1.0)
        #     .repeat(len(observation), 1)
        #     .to(observation.device)
        # )

        reward = (torch.zeros(1, 1).repeat(len(observation), 1).to(observation.device))
    
        return (value, reward, policy_logits, encoded_state)

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

    def get_weights(self):
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def normalize_encoded_state(self, encoded_state):
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
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def stack_action(self, action):
        action_one_hot = torch.nn.functional.one_hot(action.long(), self.action_space_size).float()
        return action_one_hot

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

def support_to_scalar(x):
    #.to(device=probabilities.device))
    x = torch.sign(x) * (((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))** 2 - 1)
    return x

def scalar_to_support(x):
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x
    return x
