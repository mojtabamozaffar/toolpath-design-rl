'''
Toolpath design for additive manufacturing using RL
Mojtaba Mozaffar March 2020

Significant parts of this code are adopted based on:
    https://github.com/werner-duvaud/muzero-general
    https://github.com/johan-gras/MuZero
'''

import os
import time
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environment import create_am_env
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from trainer import Trainer

class MuZeroConfig(object):
    def __init__(self):
        self.description = 'am'
        self.observation_shape = (1, 32, 32)
        self.action_space_size = 20
        self.stacked_observations = 0
        self.max_moves = 300
        self.support_size_value = 8
        self.support_size_reward = 1
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = 60
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.blocks = 2
        self.channels = 8
        self.reduced_channels_reward = 8
        self.reduced_channels_value = 8
        self.reduced_channels_policy = 8 
        self.resnet_fc_reward_layers = []
        self.resnet_fc_value_layers = []
        self.resnet_fc_policy_layers = []  
        self.value_loss_weight = 0.25
        self.n_training_loop = 100
        self.n_episodes = 20
        self.n_epochs = 400
        self.eval_episodes = 1
        self.window_size = 1000
        self.batch_size = 512
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.momentum = 0.9
        self.lr_init = 0.005
        self.lr_decay_rate = 1.0
        self.lr_decay_steps = 1000
        self.seed = 0
        self.use_last_model_value = False
        self.logdir='results/{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),self.description)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_decay = 1e-4
        self.visit_softmax_temperature_fn = lambda x: 1.0 if x<0.5*self.n_training_loop*self.n_epochs else (
                                                      0.5 if x<0.75*self.n_training_loop*self.n_epochs else 
                                                      0.25)

config = MuZeroConfig()    
np.random.seed(config.seed)
torch.manual_seed(config.seed)

os.makedirs(config.logdir, exist_ok=True)
writer = SummaryWriter(config.logdir)

env, _, env_test = create_am_env(max_steps = config.max_moves)
storage = SharedStorage(config)
replay_buffer = ReplayBuffer(config)
trainer = Trainer(storage, replay_buffer, config)
train_worker = SelfPlay(storage, replay_buffer, env, config)
test_worker = SelfPlay(storage, replay_buffer, env_test, config, test_mode=True)

hp_table = ["| {} | {} |".format(key, value) for key, value in config.__dict__.items()]
writer.add_text("Hyperparameters","| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table))

for loop in range(config.n_training_loop):
    train_worker.play(loop)
    test_worker.play(loop)
    trainer.train()

    infos = storage.get_infos()
    writer.add_scalar("1.Reward/1.Total reward", infos["total_reward"], loop)
    writer.add_scalar("1.Reward/2.Mean value", infos["mean_value"], loop)
    writer.add_scalar("2.Workers/1.Self played games",replay_buffer.get_self_play_count(),loop)
    writer.add_scalar("2.Workers/2.Training steps", infos["training_step"], loop)
    writer.add_scalar("2.Workers/3.Self played games per training step ratio", 
                      replay_buffer.get_self_play_count()/ max(1, infos["training_step"]),loop)
    writer.add_scalar("2.Workers/4.Learning rate", infos["lr"], loop)
    writer.add_scalar("3.Loss/1.Total loss", infos["total_loss"], loop)
    writer.add_scalar("3.Loss/Value loss", infos["value_loss"], loop)
    writer.add_scalar("3.Loss/Reward loss", infos["reward_loss"], loop)
    writer.add_scalar("3.Loss/Policy loss", infos["policy_loss"], loop)
    print("[{}] Reward: {:.2f}. Training step: {}/{}. Played games: {}. Loss: {:.2f}".format(
                        str(datetime.datetime.now().strftime('%H:%M:%S')),
                        infos["total_reward"],
                        loop,
                        config.n_training_loop,
                        replay_buffer.get_self_play_count(),
                        infos["total_loss"]))
