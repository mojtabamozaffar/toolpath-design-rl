import os
import time
import datetime
import numpy as np
import torch
import ray
import psutil
from torch.utils.tensorboard import SummaryWriter
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
writer = SummaryWriter(config.logdir)

model_cpu = MuZeroResidualNetwork(config)
model_cpu.eval()
init_weights = model_cpu.get_weights()
if config.use_ray:  
    num_cpus = config.num_cpus if config.num_cpus != None else psutil.cpu_count(logical=False)
    ray.init(num_cpus = num_cpus, ignore_reinit_error=True)
    storage = SharedStorage.remote(config, copy.deepcopy(init_weights))
    replay_buffer = ReplayBuffer(config)
    training_worker = Trainer.options(num_gpus=1).remote(copy.deepcopy(init_weights), storage, config)
else:
    storage = SharedStorage(config, copy.deepcopy(init_weights))
    replay_buffer = ReplayBuffer(config)
    training_worker = Trainer(copy.deepcopy(init_weights), storage, config)

report_starts = [[random.randint(0, config.observation_shape[1]-1), random.randint(0, config.observation_shape[1]-1)] for _ in range(config.eval_episodes)]
hp_table = ["| {} | {} |".format(key, value) for key, value in config.__dict__.items()]
writer.add_text("Hyperparameters","| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table))
train_id = None

for loop in range(config.n_training_loop+1):
    if config.use_ray:
        model_cpu.set_weights(copy.deepcopy(ray.get(storage.get_weights.remote())))
        temperature = config.visit_softmax_temperature_fn(ray.get(storage.get_infos.remote())["training_step"])
        game_history_ids = [play_one_game.remote(model_cpu, create_am_env, config, temperature) 
                                  for _ in range(config.n_episodes)]
        game_history_test_ids = [play_one_game.remote(model_cpu, 
                                                      partial(create_am_env_test, start_location = report_starts[i], section_id = i),
                                                      config, 
                                                      temperature = 0.0,
                                                      save= loop % 10 == 0,
                                                      filename = "toolpath_{}_{}_{}".format(loop, 'test', i))
                                  for i in range(config.eval_episodes)]
        
        game_historys = ray.get(game_history_ids)
        game_historys_test = ray.get(game_history_test_ids) 
    else:
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
      
    if config.use_ray:
        storage.set_infos.remote("total_reward", np.mean(total_rewards)) 
        buffer_id = ray.put(replay_buffer.buffer)
        batches = ray.get([get_batch.remote(buffer_id, config) for _ in range(config.n_epochs)])
        if loop > 0:
            _ = ray.get(train_id)
        train_id = training_worker.train.remote(batches)
        infos = ray.get(storage.get_infos.remote())
    else:
        storage.set_infos("total_reward", np.mean(total_rewards)) 
        batches = [get_batch(replay_buffer.buffer, config) for _ in range(config.n_epochs)]
        training_worker.train(batches)
        infos = storage.get_infos()

    # report
    if loop > 0:
        writer.add_scalar("1.Reward/1.Total reward", infos["total_reward"], loop)
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