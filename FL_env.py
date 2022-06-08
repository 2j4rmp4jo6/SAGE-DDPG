from config import for_FL as f
from FL.datasets import Dataset
from FL.attackers import Attackers
from FL.clients import Client
from FL.models import CNN_Model
from FL.groups import Groups
from FL.shuffle import Shuffle

import torch
import copy
import numpy as np
import time
import pickle

import gym

class FL_env():
    def __init__(self):
        f.device = torch.device('cuda:{}'.format(f.gpu) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')
        # 看是否使用 cuda gpu
        print(f.device)

        # 建一個 Dataset 物件
        self.my_data = Dataset()
        # 分割資料給每個 user，根據 noniid 的程度
        self.my_data.sampling()

        # 從 github 上複製來的 model
        self.FL_net = CNN_Model().to(f.device)
        # model 的參數
        self.FL_weights = self.FL_net.state_dict()

        # 建一個 Attackers 物件
        self.my_attackers = Attackers()
        if(f.attack_mode == "poison"):
            # 設定被攻擊的 label 種類
            self.my_attackers.poison_setting()
        
        # 用於之後分給 clients
        self.all_users = [i for i in range(f.total_users)]
        # 打亂所有 user 的順序
        self.idxs_users = np.random.choice(range(f.total_users), f.total_users, replace=False)

        # 建一個 Groups 物件
        self.my_groups = Groups()

        # 建一個 Shuffle 物件
        self.my_shuffle = Shuffle()
        # training + 跑 shuffle 的時間
        # 不確定這邊會不會用到，就先留著
        self.total_time = 0
        self.rl_cnt = 0
        self.rl_start = 0

        # 這裡的 agent, buffer 應該是另外放的，所以跳過

        # 從現在到程式結束的時間
        self.true_start_time = time.time()

        # 以下紀錄類不確定還會不會用到，不過也是先放著
        # total reward 紀錄
        self.total_reward = 0
        self.total_rewards = []

        # loss 紀錄
        self.threshold_rl_loss = []
        self.slicing_rl_loss = []
        
        # Attacker ratio 紀錄
        self.normal_ratio_good = []
        self.normal_ratio_bad = []
        self.attacker_ratio_bad = []
        self.attacker_ratio_good = []
        self.attacker_in_bad = []
        self.attacker_in_good = []
        self.total_attacker = []

        # state =  [FL_epoch, 三個 group 人數, intermediate group subset 數, 
        #           good group 各 label accuracy, intermediate group 各 label accuracy, bad group 各 label accuracy]
        lower_bound = [0, 0, 0, 0, 0]
        lower_bound.extend([0.0]*10*3)
        upper_bound = [f.epochs, f.total_users, f.total_users, f.total_users, f.num_clients]
        upper_bound.extend([1.0]*10*3)
        self.observation_space = gym.spaces.Box(low=np.array(lower_bound), high=np.array(upper_bound), dtype=np.float32)

        # action = [0.~0.5, 0.51~1, 1~FL.total_users]
        lower_bound = [0.0, 0.5, 1.]
        upper_bound = [0.5, 1.0, f.total_users]
        self.action_space = gym.spaces.Box(low=np.array(lower_bound), high=np.array(upper_bound),dtype=np.float32)

    
    def reset(self):
        # restart = 1 表示各 user model 收斂，一輪 RL 結束
        self.restart = 0

        # fl_epoch代表現在RL進行到第幾epoch，會在21時用RL做出決策(非隨機)之後時結束RL
        self.fl_epoch = 0

        # agent 的 action 重製應該不會放在這裡，所以也先跳過

        # total reward 重製
        self.total_reward = 0

        # 用於之後分給 clients
        self.all_users = [i for i in range(f.total_users)]
        # 打亂所有 user 的順序   
        self.idxs_users = np.random.choice(range(f.total_users), f.total_users, replace=False)

        Client.ID = 0
        self.my_clients = [Client(copy.deepcopy(self.FL_net)) for _ in range(f.num_clients + 2)]

        # 隨機 attacker ratio
        f.attack_ratio = np.random.uniform(0.1, 0.4)
        print('attacker ratio: ', f.attack_ratio)
        if(f.attack_mode == "poison"):
            # 設定被攻擊的label種類
            self.my_attackers.poison_setting()
        
        for client in self.my_clients:
            client.reset(self.fl_epoch)
        # 重置攻擊者分布
        self.my_attackers.reset()
        if(f.attack_mode == 'poison'):
        # 選定哪些為攻擊者
            self.my_attackers.choose_attackers(self.idxs_users, self.my_data)
            print("number of attacker: ", self.my_attackers.attacker_count)         
            print("all attacker: ", self.my_attackers.all_attacker)   
            print("")
        
        for client in self.my_clients:
            # 分配 users 給各 client
            # client[0], client[1] 負責放 good, bad group 的 user
            if(client.id != 0 and client.id != 1):
                self.all_users = client.split_user_to_client(self.all_users, self.my_attackers.all_attacker)
        
        observation = [self.fl_epoch, 0, f.total_users, 0, f.num_clients ]
        observation.extend([0.0]*10*3)
        return observation
    
    def step(self, round, action, agent):
        self.fl_epoch += 1

        # 每輪都要重置各 client「分到的 attackers」、「模型參數」、「模型 loss」
        for client in self.my_clients:
            client.reset(self.fl_epoch)
        # 重置 groups 中的各 client 的 acc 
        self.my_groups.reset(self.fl_epoch)
        # 重置各 groups 中的 clients
        self.my_shuffle.reset()
        
        # 開始跑方法
        start_time = time.time()

        # 分 group
        self.my_shuffle.execution_group(self.my_clients, self.my_groups, round, action[0], action[1])

        # 算 reward
        # 好的 user 被分到 good group 的數量
        good_to_good = 0
        # attacker 被分到 good group 的數量
        bad_to_good = 0        
        # attacker 被分到 bad group 的數量
        bad_to_bad = 0
        # 好的 user 被分到 bad group 的數量
        good_to_bad = 0
        for id_c in self.my_shuffle.shuffle_in_good:
            for id_u in self.my_clients[id_c].local_users:
                if id_u not in self.my_attackers.all_attacker:
                    good_to_good += 1
                else:
                    bad_to_good += 1
        for id_c in self.my_shuffle.shuffle_in_bad:
            for id_u in self.my_clients[id_c].local_users:
                if id_u in self.my_attackers.all_attacker:
                    bad_to_bad += 1
                else:
                    good_to_bad += 1
        # reward function
        reward = (good_to_good * ((1 - 0.4) / (1 - f.attack_ratio)) + bad_to_bad * (0.4 / f.attack_ratio) - good_to_bad * ((1 - 0.4) / (1 - f.attack_ratio)) - bad_to_good * (0.4 / f.attack_ratio)) * (1 ** (action[2])) - 0.5

        # 中止條件
        if round > 20 or len(self.my_groups.intermediate) == 0:
            print('--------------------End FL-------------------------')
            
            # 最後一 round reward 不給 0
            # if len(self.my_groups.intermediate) != 0:
            #     reward = 0

            # 最後一 round reward 給 0
            reward = 0
            self.total_reward += reward

            self.restart = 1

            # 更新 state (observation)
            # 原本 threshold 的部分
            # state 的定義是 [FL_epoch, 三個 group 人數, intermediate group subset 數, [good group 各 label accuracy], [intermediate group 各 label accuracy], [bad group 各 label accuracy]]
            if len(self.my_groups.acc_per_label_good) != 0:
                good = self.my_groups.acc_per_label_good
            else:
                good = [0.0] * 10
            if len(self.my_groups.acc_per_label_bad) != 0:
                bad = self.my_groups.acc_per_label_bad
            else:
                bad = [0.0] * 10
            if len(self.my_groups.acc_per_label_intermediate) != 0:
                intermediate = self.my_groups.acc_per_label_intermediate
            else:
                intermediate = [0.0] * 10
            state=[self.fl_epoch, len(self.my_groups.good), len(self.my_groups.intermediate), len(self.my_groups.bad), action[2]]
            state.extend(good)
            state.extend(intermediate)
            state.extend(bad)

            observation = state

            # 紀錄 total reward
            self.total_rewards.append(self.total_reward)
            # print('total reward: ', self.total_rewards)

            # agent 的 loss
            # print('value loss: ', agent.value_loss_record)
            # print('policy loss: ', agent.policy_loss_record)

            # 紀錄 Attacker ratio
            for idx in self.my_clients[0].local_users:
                if idx in self.my_attackers.all_attacker and idx not in self.my_clients[0].attacker_idxs:
                    self.my_clients[0].attacker_idxs.append(idx)
            for idx in self.my_clients[1].local_users:
                if idx in self.my_attackers.all_attacker and idx not in self.my_clients[1].attacker_idxs:
                    self.my_clients[1].attacker_idxs.append(idx)
            self.attacker_ratio_good.append(len(self.my_clients[0].attacker_idxs) / self.my_attackers.attacker_count)
            self.attacker_ratio_bad.append(len(self.my_clients[1].attacker_idxs) / self.my_attackers.attacker_count)
            self.normal_ratio_good.append((len(self.my_clients[0].local_users) - len(self.my_clients[0].attacker_idxs)) / (f.total_users - self.my_attackers.attacker_count))
            self.normal_ratio_bad.append((len(self.my_clients[1].local_users) - len(self.my_clients[1].attacker_idxs)) / (f.total_users -  self.my_attackers.attacker_count))
            # print('Attacker ratio good: ', self.attacker_ratio_good)
            # print('Attacker ratio bad: ', self.attacker_ratio_bad)

            path_log_variable = f.model_path + '_log_variable.txt'
            with open(path_log_variable, "wb") as file:
                pickle.dump(self.total_rewards, file)
                pickle.dump(agent.value_loss_record, file)
                pickle.dump(agent.policy_loss_record, file)
                pickle.dump(self.normal_ratio_good, file)
                pickle.dump(self.normal_ratio_bad, file)
                pickle.dump(self.attacker_ratio_bad, file)
                pickle.dump(self.attacker_ratio_good, file)

            return observation, reward, True
        
        print(self.my_attackers.attacker_num, self.my_attackers.attacker_count)

        # 不是最後一 round 的話加上去
        self.total_reward += reward

        # 更新 client、跑 shuffle
        # client ID重新計算
        Client.ID = 2
        self.my_clients = self.my_shuffle.execution_client(self.my_clients, self.my_groups, round, int(action[2]))
        print("")
        print("inter client num after shuffle", len(self.my_groups.intermediate))

        # 各 client 跑 local epoch 的時間
        # 因為實際上跑的時候是 sequence 而非 parallel
        # 要是能 parallel 更好
        self.local_ep_time = []
        self.global_test_time = 0
        print("client num", len(self.my_clients))
        print("inter client num", len(self.my_groups.intermediate))

        for client in self.my_clients:
            if len(client.local_users) != 0:
                start_ep_time = time.time()

                if(f.attack_mode == "poison"):
                    client.local_update_poison(self.my_data, self.my_attackers.all_attacker, round)
                
                end_ep_time = time.time()

                self.local_ep_time.append(end_ep_time - start_ep_time)
                
                # 對 client 進行 validation
                # 並取得所花的時間
                self.global_test_time += client.show_testing_result(self.my_data)
                
                self.my_groups.record(client)
            
        if self.my_groups.num_users_good != 0:
            self.my_groups.acc_per_label_good = [i/self.my_groups.num_users_good for i in self.my_groups.acc_per_label_good]
        if self.my_groups.num_users_bad != 0:
            self.my_groups.acc_per_label_bad = [i/self.my_groups.num_users_bad for i in self.my_groups.acc_per_label_bad]
        if self.my_groups.num_users_intermediate != 0:
            self.my_groups.acc_per_label_intermediate = [i/self.my_groups.num_users_intermediate for i in self.my_groups.acc_per_label_intermediate]
        
        # 模擬parallel，所以只取花費時間最長的local epoch time
        if len(self.local_ep_time) != 0:
            self.max_local_ep_time = self.local_ep_time[0]
            for k in range(1, len(self.local_ep_time)):
                if(self.local_ep_time[k] > self.local_ep_time[k-1]):
                    self.max_local_ep_time = self.local_ep_time[k]
        else:
            self.max_local_ep_time = 0
                   
        end_time = time.time()
        # 方法所需的時間
        shuffle_time = end_time - start_time

        print("max_local_ep_time: ", self.max_local_ep_time)
        # 跑 1 round 的時間 (validation 的時間概念上是在 central server 進行，因此不以 parallel 來算)
        round_time = self.max_local_ep_time + self.global_test_time + shuffle_time
        print("round_time: ", round_time)
        print("//////////////////////////////////////////////////////////////")

        self.total_time += round_time

        # 更新 state (observation)
        # 原本 threshold 的部分
        # state的定義是 [FL_epoch, 三個 group 人數, intermediate group subset 數, [good group 各 label accuracy], [intermediate group 各 label accuracy], [bad group 各 label accuracy]]
        if len(self.my_groups.acc_per_label_good) != 0:
            good = self.my_groups.acc_per_label_good
        else:
            good = [0.0] * 10
        if len(self.my_groups.acc_per_label_bad) != 0:
            bad = self.my_groups.acc_per_label_bad
        else:
            bad = [0.0] * 10
        if len(self.my_groups.acc_per_label_intermediate) != 0:
            intermediate = self.my_groups.acc_per_label_intermediate
        else:
            intermediate = [0.0] * 10
        state=[self.fl_epoch, self.my_groups.num_users_good, self.my_groups.num_users_intermediate, self.my_groups.num_users_bad, action[2]]
        state.extend(good)
        state.extend(intermediate)
        state.extend(bad)

        observation = state

        return observation, reward, self.restart

    def get_observation(self, epoch, action):
        # [FL_epoch, 三個 group 人數, intermediate group subset 數, 
        #  good group 各 label accuracy, intermediate group 各 label accuracy, bad group 各 label accuracy]
        if len(self.my_groups.acc_per_label_good) != 0:
            good = self.my_groups.acc_per_label_good
        else:
            good = [0.0] * 10
        if len(self.my_groups.acc_per_label_bad) != 0:
            bad = self.my_groups.acc_per_label_bad
        else:
            bad = [0.0] * 10
        if len(self.my_groups.acc_per_label_intermediate) != 0:
            intermediate = self.my_groups.acc_per_label_intermediate
        else:
            intermediate = [0.0] * 10
        observation=[self.fl_epoch, self.my_groups.num_users_good, self.my_groups.num_users_intermediate, self.my_groups.num_users_bad, action[2]]
        observation.extend(good)
        observation.extend(intermediate)
        observation.extend(bad)
        
        return observation
    
    def load(self, agent):
        path_log_variable = f.model_path + '_log_variable.txt'
        with open(path_log_variable, "rb") as file:
            self.total_rewards = pickle.load(file)
            agent.value_loss_record = pickle.load(file)
            agent.policy_loss_record = pickle.load(file)
            self.normal_ratio_good = pickle.load(file)
            self.normal_ratio_bad = pickle.load(file)
            self.attacker_ratio_bad = pickle.load(file)
            self.attacker_ratio_good = pickle.load(file)
