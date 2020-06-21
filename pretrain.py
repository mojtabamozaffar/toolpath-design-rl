import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from networks import MuZeroResidualNetwork
import matplotlib.pyplot as plt
import matplotlib
import glob
import imageio
import random
from environment import load_sections
        
class PretrainData:
    def __init__(self,config):
        self.data_size = config.data_size
        self.section_size = config.section_size
        self.window_size = config.window_size
        self.img_path = config.img_path
        self.action = config.action
        self.discount = config.discount
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []

    def generate_data(self):
        sections = load_sections(self.img_path, None)
        for _ in range(self.data_size):
            # generate observations
            section = random.choice(sections)
            H = section.shape[0]
            W = section.shape[1]
            # operate 
            # 0: fill points; 1: fill partially; 2: fill lines; 3: no operation
            operate = random.randrange(4)
            if operate == 0:
                section = self.fill_point(section, random.randrange(H*W//2))
            elif operate == 1:
                section = self.fill_partial(section)
            elif operate == 2:
                section = self.fill_line(section, random.randrange(min(H, W)//3))
            base = np.zeros((self.window_size[0]+self.section_size[0],self.window_size[1]+self.section_size[1])+(1,))
            current_point = [random.randrange(H),random.randrange(W)]
            base[self.window_size[0]//2:-self.window_size[0]//2,self.window_size[1]//2:-self.window_size[1]//2,:] = section
            window = base[current_point[0]:current_point[0]+self.window_size[0],current_point[1]:current_point[1]+self.window_size[1],:]
            window = np.transpose(window, (2, 0, 1))
            self.observations.append(window)
            
            # generate actions
            action = random.choice(self.action)
            self.actions.append(action)
            
            # generate rewards
            reward = 0
            if action >= 4:
                reward = 0
            else:
                if action == 0:
                    if current_point[0]-1 >= 0 and section[current_point[0]-1, current_point[1]] == 1:
                        reward = 1
                    else: 
                        reward = 0
                        
                if action == 1:
                    if current_point[0]+1 < H and section[current_point[0]+1, current_point[1]] == 1:
                        reward = 1
                    else: 
                        reward = 0
                        
                if action == 2:
                    if current_point[1]-1 >= 0 and section[current_point[0], current_point[1]-1] == 1:
                        reward = 1
                    else: 
                        reward = 0
                        
                if action == 3:
                    if current_point[1]+1 < W and section[current_point[0], current_point[1]+1] == 1:
                        reward = 1
                    else: 
                        reward = 0
            self.rewards.append(reward)
            
            # generate values
            n_p = np.sum(section)
            self.values.append((1-self.discount**n_p)/(1-self.discount))

    def fill_point(self, section, number):
        modified_section = section.copy()
        for _ in range(number):
            h = random.randrange(section.shape[0])
            w = random.randrange(section.shape[1])
            modified_section[h,w,0] = 0
        return modified_section

    def fill_partial(self, section):
        modified_section = section.copy()
        H=section.shape[0]
        W=section.shape[1]
        C=section.shape[2]
        direction = random.randrange(4)
        if direction == 0:
            index=random.randrange(H)
            modified_section[:index,:,:]=np.zeros((index,W,C))
        elif direction ==1:
            index=random.randrange(H)
            modified_section[index:,:,:]=np.zeros((H-index,W,C))
        elif direction == 2:
            index=random.randrange(W)
            modified_section[:,:index,:]=np.zeros((W,index,C))
        else:
            index=random.randrange(W)
            modified_section[:,index:,:]=np.zeros((W,H-index,C))
        return modified_section
            
    def fill_line(self, section, width):
        modified_section = section.copy()
        H=section.shape[0]
        W=section.shape[1]
        C=section.shape[2]
        direction = random.randrange(2)
        if direction == 0:
            index=random.randrange(H)
            if index+width<H:
                modified_section[index:index+width,:,:]=np.zeros((width,W,C))
            else:
                modified_section[index:,:,:]=np.zeros((H-index,W,C))
        else:
            index=random.randrange(W)
            if index+width<W:
                modified_section[:,index:index+width,:]=np.zeros((H,width,C))
            else:
                modified_section[:,index:,:]=np.zeros((H,H-index,C))
        return modified_section

class PreTrainer:
    def __init__(self, pretrain_network, data, config):
        self.config = config
        self.training_step = 0
        self.model = pretrain_network
        self.model.to(torch.device(config.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_init,
            weight_decay=self.config.weight_decay)
        self.data = data
        self.batch_index = 0

    def train(self):
        self.model.train()
        batch_index_list = list(np.arange(len(self.data.actions)))
        random.shuffle(batch_index_list)
        for _ in range(self.config.n_epochs):
            batch = self.get_batch(batch_index_list = batch_index_list)
            self.update_lr()
            total_loss, value_loss, reward_loss, loss_benchmark, MAE_support, MAE = self.update_weights(batch)
        
        return (self.training_step, self.optimizer.param_groups[0]["lr"], total_loss, value_loss, reward_loss, loss_benchmark, MAE_support, MAE)

    def get_batch(self, batch_index_list):
        if self.batch_index >= self.config.n_epochs:
            self.batch_index = 0
        index = self.batch_index * self.config.batch_size
        observation_batch = []
        action_batch = []
        value_batch = []
        reward_batch = []
        for i in range(self.config.batch_size):
            observation_batch.append(self.data.observations[batch_index_list[index + i]])
            action_batch.append(self.data.actions[batch_index_list[index + i]])
            value_batch.append(self.data.values[batch_index_list[index + i]])
            reward_batch.append(self.data.rewards[batch_index_list[index + i]])
        self.batch_index += 1
        return (observation_batch, action_batch, value_batch, reward_batch)


    def update_weights(self, batch):

        (observation_batch, action_batch, target_value, target_reward) = batch

        device = next(self.model.parameters()).device
        observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).float().to(device).unsqueeze(-1)
        action_batch = action_batch.unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        # observation_batch: batch, channels, heigth, width
        # action_batch: batch, 1, 1
        # target_value: batch,  
        # target_reward: batch,  
 
        target_value = self.scalar_to_support(target_value, self.config.support_size_value)
        target_reward = self.scalar_to_support(target_reward, self.config.support_size_reward)
        # target_value: batch, 2*support_size+1
        # target_reward: batch, 2*support_size+1

        # Generate predictions
        (value, reward) = self.model.pretrain_inference(observation_batch, action_batch[:,0])
        # value: batch, 2*support_size+1
        # reward: batch, 2*support_size+1

        # Compute losses
        (value_loss, reward_loss, value_loss_benchmark, reward_loss_benchmark) = self.loss_function(
            value,
            reward,
            target_value,
            target_reward,
        )

        value_loss *= self.config.loss_weight[0]
        reward_loss *= self.config.loss_weight[1]
        loss = (value_loss + reward_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        l1loss = torch.nn.L1Loss()
        MAE_value_support = l1loss(torch.softmax(value, dim = 1), target_value).mean()
        MAE_reward_support = l1loss(torch.softmax(reward, dim = 1), target_reward).mean()

        MAE_value = l1loss(self.support_to_scalar(target_value, self.config.support_size_value, 0), self.support_to_scalar(value, self.config.support_size_value)).mean()
        MAE_reward = l1loss(self.support_to_scalar(target_reward, self.config.support_size_reward, 0), self.support_to_scalar(reward, self.config.support_size_reward)).mean()

        return (loss.item(),value_loss.mean().item(),reward_loss.mean().item(), [value_loss_benchmark.mean().item(),reward_loss_benchmark.mean().item()], [MAE_value_support.item(), MAE_reward_support.item()], [MAE_value.item(), MAE_reward.item()])

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def scalar_to_support(x, support_size):
        """
        Transform a scalar to a categorical representation with (2 * support_size + 1) categories
        See paper appendix Network Architecture
        """
        # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
        x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

        # Encode on a vector
        x = torch.clamp(x, -support_size, support_size)
        floor = x.floor()
        prob = x - floor
        logits = torch.zeros(x.shape[0], 2 * support_size + 1).to(x.device)
        logits.scatter_(
            1, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
        )
        indexes = floor + support_size + 1
        prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
        indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
        logits.scatter_(1, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
        return logits

    @staticmethod
    def loss_function(
        value, reward, target_value, target_reward
    ):
        device = value.device
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1).mean()
        reward_loss = (
            (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1).mean()
        )
        log_target_value = torch.log(target_value)
        zeros = torch.zeros(target_value.shape).float().to(device)
        log_target_value = torch.where(torch.isinf(log_target_value),zeros, log_target_value)

        log_target_reward = torch.log(target_reward)
        zeros = torch.zeros(target_reward.shape).float().to(device)
        log_target_reward = torch.where(torch.isinf(log_target_reward),zeros, log_target_reward)

        value_loss_benchmark = (-target_value * log_target_value).sum(1).mean()
        reward_loss_benchmark = (-target_reward * log_target_reward).sum(1).mean()
        return value_loss, reward_loss, value_loss_benchmark, reward_loss_benchmark

    @staticmethod
    def support_to_scalar(logits, support_size, iflogits = 1):
        """
        Transform a categorical representation to a scalar
        See paper appendix Network Architecture
        """
        # Decode to a scalar
        if iflogits:
            probabilities = torch.softmax(logits, dim=1)
        else:
            probabilities = logits
        support = (
            torch.tensor([x for x in range(-support_size, support_size + 1)])
            .expand(probabilities.shape)
            .float()
            .to(device=probabilities.device)
        )
        x = torch.sum(support * probabilities, dim=1, keepdim=True)

        # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
        x = torch.sign(x) * (
            ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
            ** 2
            - 1
        )
        return x

class PretrainConfig(object):
    def __init__(self):
        # description of this test
        self.description = 'Action_4_WindowSize_32_SectionV2_Pretrain'
        # size of the data set
        self.data_size = 512*128
        # number of training loop
        self.n_training_loop = 800
        # number of train steps in single training loop
        self.n_epochs = 128
        # batch size to train in single train step
        # self.batch_size = 512
        self.batch_size = 512

        # loss weight: [value, reward]
        self.loss_weight = [1,2]
        # regularization weight
        self.weight_decay = 1e-4

        # network architecture
        # number of residual blocks
        self.blocks = 2
        # number of channels of hidden state
        self.channels = 8 
        # number of channels in networks except representation one
        self.reduced_channels = 8 
        self.resnet_fc_reward_layers = []
        self.resnet_fc_value_layers = []
        self.resnet_fc_policy_layers = []  

        # img path
        self.img_path = './Sections/Database_32x32_v2/'
        # # action label
        # self.action = [0, 1, 2, 3, 4, 5, 6, 7]
        # # number of different actions. normalized the action value to [0,1]
        # self.action_space_size = 8

        # action label
        self.action = [0, 1, 2, 3]
        # number of different actions. normalized the action value to [0,1]
        self.action_space_size = 4

        # size should be even. For establishing data set only.
        self.section_size = (32, 32)
        self.window_size = (32, 32)

        # size of the network input. in accordance with the window size
        self.observation_shape = (1, 32, 32)

        # map the value into a 2*support_size_value + 1 vector
        self.support_size_value = 24
        # map the value into a 2*support_size_reward + 1 vector
        self.support_size_reward = 1
        # reward discount
        self.discount = 0.997
        
        # optimizer parameter
        self.momentum = 0.9
        self.lr_init = 0.005
        self.lr_decay_rate = 0.97
        self.lr_decay_steps = 200
    
        # random seed
        self.seed = 0
        
        # logdir
        self.logdir='results/{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),self.description)
        # device
        self.device = "cuda: 0" if torch.cuda.is_available() else "cpu"



Pre_config = PretrainConfig()   
print(Pre_config.device)
input()
np.random.seed(Pre_config.seed)
torch.manual_seed(Pre_config.seed)

data = PretrainData(Pre_config)
data.generate_data()

pretrain_network = MuZeroResidualNetwork(Pre_config)
pretrainer = PreTrainer(pretrain_network, data, Pre_config)

writer = SummaryWriter(Pre_config.logdir)
hp_table = ["| {} | {} |".format(key, value) for key, value in Pre_config.__dict__.items()]
writer.add_text("Hyperparameters","| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table))

for loop in range(Pre_config.n_training_loop):
    (training_step, lr, total_loss, value_loss, reward_loss, loss_benchmark, MAE_support, MAE) = pretrainer.train()

    writer.add_scalar("1.Loss/1.total_loss", total_loss, loop)
    writer.add_scalar("1.Loss/2.value_loss", value_loss, loop)
    writer.add_scalar("1.Loss/3.reward_loss", reward_loss, loop)
    writer.add_scalar("1.Loss/4.net_total_loss", total_loss-loss_benchmark[0]-loss_benchmark[1], loop)
    writer.add_scalar("1.Loss/5.net_value_loss", value_loss-loss_benchmark[0], loop)
    writer.add_scalar("1.Loss/6.net_reward_loss", reward_loss-loss_benchmark[1], loop)
    writer.add_scalar("1.Loss/7.MAE_value_support", MAE_support[0], loop)
    writer.add_scalar("1.Loss/8.MAE_reward_support", MAE_support[1], loop)
    writer.add_scalar("1.Loss/9.MAE_value", MAE[0], loop)
    writer.add_scalar("1.Loss/10.MAE_reward", MAE[1], loop)
    writer.add_scalar("2.Workers/1.Training steps", training_step, loop)
    writer.add_scalar("2.Workers/2.Learning rate",lr,loop)
    print("Training step: {0}/{1}. Toal_loss: {2:.2f}. Value_loss: {3:.2f}. Reward_loss: {4:.2f}.  ".format(
                        loop,
                        Pre_config.n_training_loop,
                        total_loss,
                        value_loss,
                        reward_loss,
                        ))
state = {
    "RepresentationNetwork": pretrain_network.representation_network.state_dict(),
    "PredictValueNetwork": pretrain_network.prediction_network.state_dict(),
    "PredictRewardNetwork": pretrain_network.dynamics_network.state_dict(),
}
torch.save(state, Pre_config.description +".pth.tar")



