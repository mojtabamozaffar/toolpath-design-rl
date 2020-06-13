import torch

from networks import MuZeroResidualNetwork

class SharedStorage:
    def __init__(self, config):
        self.config = config
        self.network = MuZeroResidualNetwork(config)
        self.network.to(torch.device(config.device))
        self.network.train()
        self.network_cpu = MuZeroResidualNetwork(config)
        self.network_cpu.eval()
        self.infos = {
            "total_reward": 0,
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
