import os
import time
import datetime
import numpy as np
import torch
import ray
import psutil
from functools import partial
import random
import copy

import global_config as config
from environment import create_am_env, create_am_env_test
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer, get_batch
from self_play import play_one_game
from trainer import Trainer
from networks import MuZeroResidualNetwork

os.environ["CUDA_VISIBLE_DEVICES"]="0"
random.seed(0)
os.makedirs(config.logdir, exist_ok=True)

model_cpu = MuZeroResidualNetwork(config)
model_cpu.eval()
model_cpu.set_weights(torch.load('results/test_model.pt'))
init_weights = model_cpu.get_weights()

storage = SharedStorage(config, copy.deepcopy(init_weights))
replay_buffer = ReplayBuffer(config)
training_worker = Trainer(copy.deepcopy(init_weights), storage, config)

report_starts = [[random.randint(0, config.observation_shape[1]-1), random.randint(0, config.observation_shape[1]-1)] for _ in range(config.eval_episodes)]

for loop in range(config.n_training_loop+1):
    model_cpu.set_weights(copy.deepcopy(storage.get_weights()))
    temperature = config.visit_softmax_temperature_fn(storage.get_infos()["training_step"])
    game_historys = [play_one_game(model_cpu, create_am_env, config, temperature) 
                              for _ in range(config.n_episodes)]
    game_historys_test = [play_one_game(model_cpu, 
                                                  partial(create_am_env_test, start_location = report_starts[i], section_id = i),
                                                  config, 
                                                  temperature = 0.0,
                                                  save= loop % 10 == 0,
                                                  filename = "toolpath_{}_{}_{}".format(loop, 'test', i))
                             for i in range(config.eval_episodes)]
    
    for game_history in game_historys:
        replay_buffer.save_game(game_history)
    total_rewards = []
    for game_history in game_historys_test:
        total_rewards.append(sum(game_history.reward_history))
      
    storage.set_infos("total_reward", np.mean(total_rewards)) 
    batches = [get_batch(replay_buffer.buffer, config) for _ in range(config.n_epochs)]
    training_worker.train(batches)
    infos = storage.get_infos()

    # report
    if loop > 0:
        print("[{}] Reward: {:.2f}. Training step: {}/{}. Played games: {}. Loss: {:.2f}".format(
                            str(datetime.datetime.now().strftime('%H:%M:%S')),
                            infos["total_reward"],
                            loop,
                            config.n_training_loop,
                            replay_buffer.get_self_play_count(),
                            infos["total_loss"]))