import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import glob
import imageio
import gym
from gym import spaces
import random
import cv2
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class ToolpathEnvironmentGym(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, sections, start, max_steps = 500):
        # self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self.actions = [0, 1, 2, 3]
        self.spec = DummyEnvSpec('toolpath_env_gym_v0')
        self.action_space = spaces.Discrete(len(self.actions))
        self.im_ind = 0
        self.section_size = sections[0].shape[0]
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(1, self.section_size, self.section_size), 
                                            dtype=np.uint8)
        self.init_start = start
        self.sections = sections
        self.base = random.choice(sections).copy()
        self.filled = np.zeros(shape=(self.section_size, self.section_size, 1), dtype=np.uint8)
        self.max_steps = max_steps
        self.step_num = 0
        if self.init_start == 'random':
            self.laser_loc = [random.randint(0,self.section_size-1), random.randint(0,self.section_size-1)]
        else:
            self.laser_loc = [self.init_start[0], self.init_start[1]]
        self.laser = False
        self.is_terminal = False
        self.viewer = None
        self.window_size = 32
        
        self.action_history = []
        self.toolpath_x = [self.laser_loc[0]]
        self.toolpath_y = [self.laser_loc[1]]

    def reset(self):
        self.base = random.choice(self.sections).copy()
        self.filled = np.zeros(shape=(self.section_size, self.section_size, 1), dtype=np.uint8)
        if self.init_start == 'random':
            self.laser_loc = [random.randint(0,self.section_size-1), random.randint(0,self.section_size-1)]
        else:
            self.laser_loc = [self.init_start[0], self.init_start[1]]
        self.toolpath_x = [self.laser_loc[0]]
        self.toolpath_y = [self.laser_loc[1]]
        self.step_num = 0
        self.is_terminal = False
        self.laser = False
        self.action_history = []
        
        return self._get_observation()
    
    def step(self, action):
        old_filled = self.filled.copy()
        if action in self._valid_actions():
            # change laser_loc and laser
            if action == 0:
                self.laser_loc[0] -= 1
                self.laser = True
            elif action == 1:
                self.laser_loc[0] += 1
                self.laser = True
            elif action == 2:
                self.laser_loc[1] += 1
                self.laser = True
            elif action == 3:
                self.laser_loc[1] -= 1
                self.laser = True
            # elif action == 4:
            #     self.laser_loc[0] -= 1
            #     self.laser = False
            # elif action == 5:
            #     self.laser_loc[0] += 1
            #     self.laser = False
            # elif action == 6:
            #     self.laser_loc[1] += 1
            #     self.laser = False
            # elif action == 7:
            #     self.laser_loc[1] -= 1
            #     self.laser = False
        
        # change filled
        if self.laser:
            self.filled[self.laser_loc[0],self.laser_loc[1], 0] = 1

        self.action_history.append(action)
        
        # change is_terminal
        if self._is_successful() or self.step_num >= self.max_steps:
            self.is_terminal = True
            
        if action in self._valid_actions():
            reward = task(self.base, old_filled, self.laser_loc, self.action_history, self.is_terminal)
        else:
            reward = 0.0
                                
        info = ''
        self.step_num +=1
        self.toolpath_x.append(self.laser_loc[0])
        self.toolpath_y.append(self.laser_loc[1])
        
        return (self._get_observation(), reward , self.is_terminal, info)
    
    def render(self, mode='human', close=False):
        if self.viewer == None:
            self.viewer = ToolpathVisualizer()   
        return self.viewer.render(self._get_rgb(), return_rgb_array = mode=='rgb_array')
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def _get_rgb(self):
        laserOn_cl = np.array([255, 0, 0], dtype = np.uint8)
        laserOff_cl = np.array([146, 77, 77], dtype = np.uint8)
        remain_cl = np.array([[0, 102, 204]], dtype = np.uint8)
        filled_cl = np.array([[0, 153, 0]], dtype = np.uint8)
        
        rgb = 255* np.ones((self.section_size,self.section_size,3), dtype=np.uint8)
        rgb = self._mask(rgb, self.base, remain_cl)
        rgb = self._mask(rgb, self.filled, filled_cl)
        if self.laser:
            rgb[self.laser_loc[0], self.laser_loc[1], :] = laserOn_cl
        else:
            rgb[self.laser_loc[0], self.laser_loc[1], :] = laserOff_cl
        return rgb
    
    def _get_observation(self):
        observation = self.base.copy().astype(np.int32) - self.filled
        observation = np.clip(observation, a_min = 0, a_max=1)
        observation = _create_window(self.window_size, observation, self.laser_loc)
        # observation[self.laser_loc[0], self.laser_loc[1], 0] = -1
        observation = np.transpose(observation, (2, 0, 1))
        return observation.astype(np.float32)
    
    def _mask(self, base, mask, color):
        mask = mask.astype(bool)
        mask = mask.repeat(3, axis=2)
        color_base = np.ones(shape=(base.shape[0],base.shape[1],1), dtype=np.uint8)
        color_base = color_base.dot(color)
        new_base = base.copy()
        new_base[mask] = color_base[mask]
        return new_base
        
    def _is_successful(self):
        fill_check = np.greater_equal(self.filled, self.base)
        return False if False in fill_check else True
    
    def _valid_actions(self):
        actions = self.actions.copy()
        if self.laser_loc[1] == 0:
            actions.remove(3)
            # actions.remove(7)
        if self.laser_loc[1] == self.section_size-1:
            actions.remove(2)
            # actions.remove(6)
        if self.laser_loc[0] == 0:
            actions.remove(0)
            # actions.remove(4)
        if self.laser_loc[0] == self.section_size-1:
            actions.remove(1)
            # actions.remove(5)
        return actions
    
    def plot_state(self, save = False, folder='', filename='state'):
        fig, ax = plt.subplots()
        ax.imshow(self._get_rgb())
        if save:
            fig.savefig(folder + '/' + filename +'.png')
            plt.close()
    
    def plot_toolpath(self, save = False, folder='', filename='toolpath'):
        cmap = matplotlib.cm.get_cmap('cool')
        colors = cmap(np.linspace(0,1, len(self.toolpath_x)-1))
        fig, ax = plt.subplots()
        ax.tick_params(axis='both',which='both', bottom=False,top=False,left = False, right=False, labelbottom=False, labelleft=False)
        ax.imshow(self.base.reshape(self.section_size,self.section_size), cmap='binary', norm = matplotlib.colors.Normalize(vmin=0.0, vmax=5.0))
        if len(self.toolpath_x)>2:
            ax.plot(self.toolpath_y[0], self.toolpath_x[0], marker = 'D', color = colors[0])
            for i in range(len(self.toolpath_x)-1):
                ax.plot(self.toolpath_y[i:i+2], self.toolpath_x[i:i+2], color=colors[i])
            if self.toolpath_x[-1]==self.toolpath_x[-2] and self.toolpath_y[-1]==self.toolpath_y[-2]:
                ax.plot(self.toolpath_y[-1], self.toolpath_x[0-1], marker = 'D', color = colors[-1])
            else:
                ax.arrow(self.toolpath_y[-2],self.toolpath_x[-2], self.toolpath_y[-1]-self.toolpath_y[-2], self.toolpath_x[-1]-self.toolpath_x[-2], head_width=0.75, head_length=0.75, fc=colors[-1], ec=colors[-1], length_includes_head=True, alpha=1.0)
        ax.set_xlim(0,self.section_size)
        ax.set_ylim(0,self.section_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        if save:
            fig.savefig(folder + '/' + filename +'.png', dpi=300)
            plt.close()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def to_play(self):
        return 0
    
def task(base, old_filled, laser_loc, action_his, finished):
    if base[laser_loc[0],laser_loc[1], 0] == 1 and old_filled[laser_loc[0],laser_loc[1], 0] == 0 and action_his[-1]<4:
        return 1.0
    # elif action_his[-1] > 3:
    #     return 0.1
    else:
        return 0.0
    
class ToolpathVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.legend_data = [[0,[0,102,204],'section'],
                       [1,[0,153,0],'filled'],
                       [2,[255,0,0],'laser on'],
                       [3,[146,77,77],'laser off']]
        self.handles = [Rectangle((0,0),1,1, color = tuple((v/255 for v in c))) for k,c,n in self.legend_data]
        self.labels = [n for k,c,n in self.legend_data]
        plt.show(block=False)
        
    def render(self, rgb,return_rgb_array=False):
        self.ax.imshow(rgb)

        self.ax.legend(self.handles,self.labels)
        if return_rgb_array:
#            plt.pause(0.0001)
            return cv2.resize(rgb, dsize=(800, 800), interpolation=cv2.INTER_NEAREST) 
        else:
            plt.pause(0.0001)
            return None
        
    def close(self):
        plt.close()
        
class DummyEnvSpec:
    def __init__(self, id):
        self.id = id
        
def _create_window(window_size, base, location):
    padded_base = np.zeros(shape = (base.shape[0]+window_size, base.shape[0]+window_size, 1), dtype=np.uint8)
    padded_base[window_size//2:-window_size//2:, window_size//2:-window_size//2:, :] = base
    window = padded_base[location[0]:location[0]+window_size, location[1]:location[1]+window_size,:]
    return window
    
    
def load_sections(img_path, sample_number):
    sections = []
    if not sample_number == None:
        img = np.asarray(imageio.imread(glob.glob(img_path+str(sample_number)+'.png')[0]))/255
        img = img.astype(np.uint8)
        sections.append(img.reshape(img.shape+(1,)))
    else:
        for path in glob.glob(img_path+'*.png'):
            img = np.asarray(imageio.imread(path))/255
            img = img.astype(np.uint8)
            sections.append(img.reshape(img.shape+(1,)))
        random.shuffle(sections)
    return sections

def create_am_env(max_steps = 100, img_path = 'Sections/Database_32x32/', start_location = 'random', section_id = None):
    section = load_sections(img_path, section_id)
    env = ToolpathEnvironmentGym(section, start_location, max_steps = max_steps)
    return env

def create_am_env_test(max_steps = 100, img_path = 'Sections/Database_32x32/Report/', start_location = 'random', section_id = None):
    section = load_sections(img_path, section_id)
    env = ToolpathEnvironmentGym(section, start_location, max_steps = max_steps)
    return env
