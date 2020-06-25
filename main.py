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
import threading

from environment import create_am_env, create_am_env_test
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer, get_batch
from self_play import play_one_game
from trainer import Trainer

class MuZeroConfig(object):
    def __init__(self):
        self.description = 'AM'
        self.observation_shape = (1, 32, 32)
        self.action_space_size = 4
        self.max_moves = 400
        self.support_size_value = 20
        self.support_size_reward = 1
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = 300
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 1000
        self.pb_c_init = 1.25
        self.blocks = 2
        self.channels = 8 
        self.reduced_channels = 8 
        self.resnet_fc_reward_layers = []
        self.resnet_fc_value_layers = []
        self.resnet_fc_policy_layers = []  
        self.value_loss_weight = 0.25
        self.n_training_loop = 100
        self.n_episodes = 20
        self.n_epochs = 400
        self.eval_episodes = 10
        self.window_size = 1000
        self.batch_size = 512
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.momentum = 0.9
        self.lr_init = 0.005
        self.lr_decay_rate = 1.0
        self.lr_decay_steps = 1000
        self.logdir='results/{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),self.description)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.weight_decay = 1e-4
        # self.num_cpus = 10
        self.num_cpus = 12
        self.visit_softmax_temperature_fn = lambda x: 1.0 if x<0.5*self.n_training_loop*self.n_epochs else (
                                                      0.5 if x<0.75*self.n_training_loop*self.n_epochs else 
                                                      0.25)

config = MuZeroConfig()    
random.seed(0)
num_cpus = config.num_cpus if config.num_cpus != None else psutil.cpu_count(logical=False)
ray.init(num_cpus = num_cpus, ignore_reinit_error=True)
os.makedirs(config.logdir, exist_ok=True)
writer = SummaryWriter(config.logdir)
storage = SharedStorage(config)
# storage.load_pretrain_param("./pretrain_action_4.pth.tar")
replay_buffer = ReplayBuffer(config)
trainer = Trainer(storage, config)
report_starts = [[random.randint(0,31), random.randint(0,31)] for _ in range(config.eval_episodes)]

hp_table = ["| {} | {} |".format(key, value) for key, value in config.__dict__.items()]
writer.add_text("Hyperparameters","| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table))

class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            print('Thread is not finished.')
            return None

def play_games(loop):
    model = storage.network_cpu
    model.set_weights(storage.network.get_weights())
    
    # self-play training
    temperature = config.visit_softmax_temperature_fn(storage.get_infos()["training_step"])
    
    game_history_ids = [play_one_game.remote(model, create_am_env, config, temperature) 
                              for i in range(config.n_episodes)]
    if loop % 10 == 0:
        game_history_test_ids = [play_one_game.remote(model, 
                                                      partial(create_am_env_test, start_location = report_starts[i], section_id = i), 
                                                      config,
                                                      temperature=0.0,
                                                    #   save=True,
                                                    #   filename = "toolpath_{}_{}_{}".format(loop, 'test', i)
                                                      )
                            for i in range(config.eval_episodes)]
    else:
        game_history_test_ids = [play_one_game.remote(model, 
                                                      partial(create_am_env_test, start_location = report_starts[i], section_id = i), 
                                                      config,
                                                      temperature=0.0)
                            for i in range(config.eval_episodes)]
    game_historys = ray.get(game_history_ids)
    game_historys_test = ray.get(game_history_test_ids)  
    for game_history in game_historys:
        replay_buffer.save_game(game_history)
    total_rewards = []
    for game_history in game_historys_test:
        total_rewards.append(sum(game_history.reward_history))
    storage.set_infos("total_reward", np.mean(total_rewards))
    
    # get batches
    buffer_id = ray.put(replay_buffer.buffer)
    batches = ray.get([get_batch.remote(buffer_id, config) for _ in range(config.n_epochs)])
    return batches


def train_networks(batches):
    trainer.train(batches)

batches = []

for loop in range(config.n_training_loop + 1):
    time0 = time.time()
    thread_play = MyThread(play_games, args = (loop, ))
    thread_play.start()
    if loop == 0:
        thread_play.join()
        batches = thread_play.get_result()
    else:
        thread_train = MyThread(train_networks, args = (batches, ))
        thread_train.start()
        thread_train.join()
        thread_play.join()
        batches = thread_play.get_result()
    
    time1 = time.time()
    print(time1-time0)

    infos = storage.get_infos()
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

ray.shutdown()