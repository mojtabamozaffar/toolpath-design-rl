import torch
import numpy as np
import ray
import global_config

class SharedStorage:
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        self.infos = {
            "total_reward": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
        }

    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights
        
    def get_infos(self):
        return self.infos

    def set_infos(self, key, value):
        self.infos[key] = value
        
if global_config.use_ray:
    SharedStorage = ray.remote(SharedStorage)