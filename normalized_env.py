
from email.headerregistry import Group
import gym
import numpy as np
import config.for_FL as FL
from  FL import groups, clients
from FL.models import CNN_Model

import torch

FL.device = torch.device('cuda:{}'.format(FL.gpu) if torch.cuda.is_available() and FL.gpu!=-1 else 'cpu')

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self, env) :
        super().__init__(env)

        # state =  [FL_epoch, 三個 group 人數, intermediate group subset 數, 
        #           good group 各 label accuracy, intermediate group 各 label accuracy, bad group 各 label accuracy]
        lower_bound = [0, 0, 0, 0, 0]
        lower_bound.extend([0.0]*10*3)
        upper_bound = [FL.epochs, FL.total_users, FL.total_users, FL.total_users, FL.num_clients]
        upper_bound.extend([1.0]*10*3)
        self.observation_space = gym.spaces.Box(low=np.array(lower_bound), high=np.array(upper_bound), dtype=np.float32)

        # action = [0.~0.5, 0.51~1, 1~FL.total_users]
        lower_bound = [0.0, 0.0, 0.0]
        upper_bound = [1.0, 1.0, FL.total_users]
        self.action_space = gym.spaces.Box(low=np.array(lower_bound), high=np.array(upper_bound),dtype=np.float32)

        # update observation要用
        self.groups = groups.Groups()
        # claculate reward 要用
        FL_net = CNN_Model().to(FL.device)
        FL_net.state_dict()
        self.clients = [clients.Client(FL_net) for _ in range(FL.num_clients + 2)]

    def step(self, epoch, action):
        # 這邊先寫死
        observation =[epoch, 0, 0, 0, 0]
        observation.extend([0.0]*10*3)
        reward = 0
        done=False

        return observation, reward, done, None

    def reset(self, epoch) :
        # 重置 FL env
        
        observation = [epoch, 0, FL.total_users, 0, 10]
        observation.extend([0.0]*10*3)
        return observation

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

    def get_observation(self, epoch, action):
        # [FL_epoch, 三個 group 人數, intermediate group subset 數, 
        #  good group 各 label accuracy, intermediate group 各 label accuracy, bad group 各 label accuracy]
        if len(self.groups.acc_per_label_good) != 0:
            good = self.groups.acc_per_label_good
        else:
            good = [0.0] * 10
        if len(self.groups.acc_per_label_bad) != 0:
            bad = self.groups.acc_per_label_bad
        else:
            bad = [0.0] * 10
        if len(self.groups.acc_per_label_intermediate) != 0:
            intermediate = self.groups.acc_per_label_intermediate
        else:
            intermediate = [0.0] * 10
        observation=[epoch, len(self.groups.good), len(self.groups.intermediate), len(self.groups.bad), action[2]]
        observation.extend(good)
        observation.extend(intermediate)
        observation.extend(bad)
        
        return observation
