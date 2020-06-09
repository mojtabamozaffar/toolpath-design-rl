import torch

from networks import MuZeroFullyConnectedNetwork, MuZeroResidualNetwork

class SharedStorage:
    def __init__(self, config):
        self.config = config
        # self.current_network = MuZeroFullyConnectedNetwork(config)
        self.current_network = MuZeroResidualNetwork(config)
        self.infos = {
            "total_reward": 0,
            "player_0_reward": 0,
            "player_1_reward": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
        }

    def get_infos(self):
        return self.infos

    def set_infos(self, key, value):
        self.infos[key] = value
