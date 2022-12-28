
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

import config.for_FL as FL
# from ipdb import set_trace as debug

criterion = nn.MSELoss()

FL.device = torch.device('cuda:{}'.format(FL.gpu) if torch.cuda.is_available() and FL.gpu != -1 else 'cpu')

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

        # 紀錄 loss
        self.value_loss_record = []
        self.policy_loss_record = []

        # 超過 boundary
        self.over_boundary = 0

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        self.value_loss_record.append(value_loss.data.item())
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        self.policy_loss_record.append(policy_loss.data.item())
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        '''
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
        '''
        self.actor.to(FL.device)
        self.actor_target.to(FL.device)
        self.critic.to(FL.device)
        self.critic_target.to(FL.device)
        print("model: ", next(self.actor.parameters()).device)

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = []
        action = np.append(action, [np.random.randint(FL.num_clients + 1)])
        action = np.append(action, [np.random.randint(FL.num_clients + 1)])
        print("rand action: ", action)
        if action[0] + action[1] > 10:
            action = [10, 0]
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        self.over_boundary = 0

        action = to_numpy(
            self.actor(to_tensor(np.array(s_t)))
        )#.squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        # slicing_action = round(action[2])
        # if slicing_action < 1:
        #     slicing_action = 1
        
        if action[0] > 1:
            self.over_boundary += action[0] - 1
            print("out of boundary action[0]: ", action[0])
        elif action[0] < 0:
            self.over_boundary += 0 - action[0]
            print("out of boundary action[0]: ", action[0])
        
        if action[1] > 1:
            self.over_boundary += action[1] - 1
            print("out of boundary action[1]: ", action[1])
        elif action[1] < 0:
            self.over_boundary += 0 - action[1]
            print("out of boundary action[1]: ", action[1])
        
        if(self.over_boundary != 0):
            print("action before clip: ", action)

        low = 0
        high = FL.num_clients

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action[0] = action[0] * scale_factor + reloc_factor
        action[0] = round(np.clip(action[0], low, high))

        low = 0
        high = FL.num_clients

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action[1] = action[1] * scale_factor + reloc_factor
        action[1] = round(np.clip(action[1], low, high))

        print("selected action: ", action)

        if action[0] + action[1] > 10:
            action[0] = 10
            action[1] = 0

        action = np.append(action[0], action[1])
        
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        path_actor = FL.model_path + '_actor.pt'
        self.actor.load_state_dict(torch.load(path_actor))
        path_actor_target = FL.model_path + '_actor_target.pt'
        self.actor_target.load_state_dict(torch.load(path_actor_target))
        path_actor_optim = FL.model_path + '_actor_optim.pt'
        self.actor_optim.load_state_dict(torch.load(path_actor_optim))
        path_critic = FL.model_path + '_critic.pt'
        self.critic.load_state_dict(torch.load(path_critic))
        path_critic_target = FL.model_path + '_critic_target.pt'
        self.critic_target.load_state_dict(torch.load(path_critic_target))
        path_critic_optim = FL.model_path + '_critic_optim.pt'
        self.critic_optim.load_state_dict(torch.load(path_critic_optim))

    def save_model(self,output):
        path_actor = FL.model_path + '_actor.pt'
        torch.save(self.actor.state_dict(), path_actor)
        path_actor_target = FL.model_path + '_actor_target.pt'
        torch.save(self.actor_target.state_dict(), path_actor_target)
        path_actor_optim = FL.model_path + '_actor_optim.pt'
        torch.save(self.actor_optim.state_dict(), path_actor_optim)
        path_critic = FL.model_path + '_critic.pt'
        torch.save(self.critic.state_dict(), path_critic)
        path_critic_target = FL.model_path + '_critic_target.pt'
        torch.save(self.critic_target.state_dict(), path_critic_target)
        path_critic_optim = FL.model_path + '_critic_optim.pt'
        torch.save(self.critic_optim.state_dict(), path_critic_optim)

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

    def calc_reward(self, env, action):
        user_point = 0.
        for client in env.clients:
            if client.id != 0 and client.id != 1:
                if client.acc_per_label_min > action[1]:
                    user_point += len(client.local_users) * (1 - 0.4) / (1 - FL.attack_ratio)
                elif client.acc_per_label_min < action[0]:
                    user_point += len(client.local_users) * 0.4 / FL.attack_ratio
        user_point = user_point * (0.9**action[2])
        return user_point
