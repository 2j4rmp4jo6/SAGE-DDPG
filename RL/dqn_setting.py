#!/usr/bin/env python3
import numpy as np
import collections
import torch
import torch.nn as nn

#from tensorboardX import SummaryWriter

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'a_round', 'new_state'])

GAMMA = 0.9

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])   #解壓縮
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class thresholdAgent:
    def __init__(self, exp_buffer):
        self.exp_buffer = exp_buffer
        self.total_reward = 0
        # 新增以下，action預設[0.0, 0.9]
        self.action = [0.01, 0.9]
        self.state = None
        self.old_state = None

    # def reset_total_reward(self):
    #     self.total_reward = 0

    def get_total_reward(self):
        return self.total_reward

    def choose_action(self, state, net, fl_epoch, epsilon=0.0, device="cpu"):
        action_list = []
        b = np.arange(0.01, 0.51, 0.05).tolist()
        g = np.arange(0.50, 1, 0.05).tolist()
        b = [round(num, 2) for num in b]
        g = [round(num, 2) for num in g]
        for i in range(0, 10):
            for j in range(0, 10):
                action_list.append([b[i], g[j]])
        if fl_epoch < 5:
            self.action =  action_list[np.random.choice(100)]
            self.action[0] = 0.01
        elif np.random.random() < epsilon:
            # 隨機選動作(通常是因為RL model還沒跑足夠多局數)
            # action = np.random.choice([i for i in range(2)])
            print("random action")
            self.action =  action_list[np.random.choice(100)]
            # self.action[0] = 0.0
        else:
            # 根據RL model決定動作
            # 待改
            state_a = np.array(state, copy=False)
            state_v = torch.from_numpy(state_a).to(device)
            q_vals_v = net(state_v)
            # print(q_vals_v)
            _, act_v = torch.max(torch.unsqueeze(q_vals_v,0),dim=1)
            self.action = action_list[act_v.item()] # index

            #不確定
            # self.action.append(act_v.item())
            
        # return action


    def get_reward(self, state, new_state, a_round, clients, threshold, action, not_final_rond):
        # 也就是acc per label list中最差的acc要有所進步才能得到reward
        # if(np.min(new_state[5])>np.min(state[5])):
        #     reward = 1
        # else:
        #     reward = 0

        # change group min label acc
        acc_sum = 0
        num_user = 0
        reward = 0
        for client in clients:
            if client.id != 0 and client.id != 1:
                if client.acc_per_label_min > action[1]:
                    acc_sum = acc_sum + client.acc_per_label_min * len(client.local_users)
                    num_user += len(client.local_users)
        if num_user != 0:
            reward = acc_sum / num_user

        # 更新total reward
        self.total_reward = GAMMA * self.total_reward + reward
        print('threshold total reward:',self.total_reward)

        ###################################################################################
        # 結束之後再給reward
        # if not_final_rond:
        #     reward = 0
        # else:
        #     reward = self.total_reward

        # self.total_reward = reward
        # if not_final_rond:
        #     print('threshold reward:',reward)
        # else:
        #     print('threshold total reward:',self.total_reward)
        ###################################################################################
        print('threshold reward:',reward)
        # print('threshold state:',state)
        print('threshold new_state:',new_state) 
        # exp = Experience(state, new_state[4], reward, a_round, new_state)
        exp = Experience(state, action, reward, a_round, new_state)
        

        # 我沒記錯的話，應該是buffer存夠局數會更新一次RL model
        self.exp_buffer.append(exp)

class sliceAgent:
    def __init__(self, exp_buffer):
        self.exp_buffer = exp_buffer
        self.total_reward = 0
        # 新增以下，action預設10
        self.action = 10
        self.state = None
        self.old_state = None

    # def reset_total_reward(self):
    #     self.total_reward = 0

    def get_total_reward(self):
        return self.total_reward

    def choose_action(self, state, net, fl_epoch, epsilon=0.0, device="cpu"):
        # action_list = [i for i in range(1,101)]
        action_list = [i for i in range(1,51)]
        if np.random.random() < epsilon:
            # 隨機選動作(通常是因為RL model還沒跑足夠多局數)
            # action = np.random.choice([i for i in range(2)])
            print("random action")
            # self.action =  action_list[np.random.choice(100)]
            self.action =  action_list[np.random.choice(50)]
        else:
            # 根據RL model決定動作
            # 待改
            state_a = np.array(state, copy=False)
            state_v = torch.from_numpy(state_a).to(device)
            q_vals_v = net(state_v)
            # print(q_vals_v)
            _, act_v = torch.max(torch.unsqueeze(q_vals_v,0),dim=1)
            self.action = action_list[act_v.item()] # index
        
        # 鎖定action
        self.action = 10

    def get_reward(self, state, new_state, a_round, clients, threshold, action, not_final_rond):
        num_user = 0
        reward = 0
        for client in clients:
            if client.id != 0 and client.id != 1:
                if client.acc_per_label_min > threshold[1]:
                    num_user += len(client.local_users)
                elif client.acc_per_label_min < threshold[0]:
                    num_user += len(client.local_users)
        if action != 0:
            reward = num_user / action

        # 更新total reward
        self.total_reward = GAMMA * self.total_reward + reward
        print('slice total_reward:', self.total_reward)
        
        ###################################################################################
        # 結束之後再給reward
        # if not_final_rond:
        #     reward = 0
        # else:
        #     reward = self.total_reward
        ###################################################################################

        # self.total_reward = reward
        print('slice_reward:',reward)
        # print('slice_state:',state)
        print('slice_new_state:',new_state) 
        exp = Experience(state, action, reward, a_round, new_state)

        # 我沒記錯的話，應該是buffer存夠局數會更新一次RL model
        self.exp_buffer.append(exp)


def calc_threshold_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    idx_action = []
    idx = []
    b = np.arange(0.01, 0.51, 0.05).tolist()
    g = np.arange(0.50, 1, 0.05).tolist()
    b = [round(num, 2) for num in b]
    g = [round(num, 2) for num in g]
    for i in range(0, 10):
        for j in range(0, 10):
            idx.append([b[i], g[j]])
    # print("actions_v",actions_v)
    for i in actions:
        idx_action.append(idx.index(i.tolist()))  
  
    state_action_values = net(states_v).gather(1, torch.tensor(idx_action).to(device).unsqueeze(-1)).squeeze(-1)
    # state_action_values =state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    next_state_values = tgt_net(next_states_v).max(1)[0]
    #next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    
    print('threshold state_action_values:',state_action_values)
    print('threshold expected_state_action_values:',expected_state_action_values)

    # Mean Square Error
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_slicing_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    idx_action = []
    idx = [i for i in range(1, 51)]
    # idx = [i for i in range(1,101)]

    # print("actions_v",actions_v)
    for i in actions:
        idx_action.append(idx.index(i)) 
  
    state_action_values = net(states_v).gather(1, torch.tensor(idx_action).to(device).unsqueeze(-1)).squeeze(-1)
    # state_action_values =state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    next_state_values = tgt_net(next_states_v).max(1)[0]
    #next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    
    print('slicing state_action_values:',state_action_values)
    print('slicing expected_state_action_values:',expected_state_action_values)

    # Mean Square Error
    return nn.MSELoss()(state_action_values, expected_state_action_values)
