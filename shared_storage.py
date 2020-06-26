import torch
import numpy as np
import ray

@ray.remote
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
    
# TO DO: needs to be updated to be compatible
    def load_pretrain_param(self, param_path):
        pass
#         pretrain_state = torch.load(param_path)
#         repre_state = self.network.representation_network.state_dict()
#         Pre_repre = {k: v for k, v in pretrain_state["RepresentationNetwork"].items() if k in repre_state}
#         repre_state.update(Pre_repre)
#         self.network.representation_network.load_state_dict(repre_state)
#         dynam_state = self.network.dynamics_network.state_dict()
#         Pre_dynam = {k: v for k, v in pretrain_state["PredictRewardNetwork"].items() if k in dynam_state}
#         dynam_state.update(Pre_dynam)
#         self.network.dynamics_network.load_state_dict(dynam_state)
#         predict_state = self.network.prediction_network.state_dict()
#         Pre_predict = {k: v for k, v in pretrain_state["PredictValueNetwork"].items() if k in predict_state}
#         predict_state.update(Pre_predict)
#         self.network.prediction_network.load_state_dict(predict_state)
        
    def get_infos(self):
        return self.infos

    def set_infos(self, key, value):
        self.infos[key] = value
