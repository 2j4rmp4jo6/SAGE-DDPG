from .dqn_setting import ExperienceBuffer, sliceAgent, thresholdAgent
from .dqn_model import DQN
from ..config import for_FL as f
from ..config import for_RL as r

import torch.optim as optim
import numpy as np

class RL_setting():

    def __init__(self):
        self.threshold_state = []
        self.threshold_action = None
        self.threshold_net = None
        self.threshold_target_net = None
        self.threshold_buffer = None
        self.threshold_agent = None
        self.threshold_epsilon = None
        self.threshold_optimizer = None
        self.threshold_count = 0

        self.slicing_state = []
        self.slicing_action = None
        self.slicing_net = None
        self.slicing_target_net = None
        self.slicing_buffer = None
        self.slicing_agent = None
        self.slicing_epsilon = None
        self.slicing_optimizer = None
        self.slicing_count = 0

    
    def for_threshold_dqn(self):
        # # state的定義是 [good group各model的平均acc，bad group中各model的平均acc，good group中最差的acc，bad group中最差的acc，該model現在在哪個group中]  
        # self.state = [i for i in range(5)]  
        # # 除上述，還加上該model validation後，得到的各類別圖片的acc (一個list)
        # self.state.extend([i for i in range(10)])
        self.threshold_buffer = ExperienceBuffer(r.REPLAY_SIZE)
 
        # state的定義是 [FL_epoch, 三個 group 人數, intermediate group subset 數, [good group 各 label accuracy], [intermediate group 各 label accuracy], [bad group 各 label accuracy]]  
        substate = [i for i in range(10)] 
        self.threshold_state = [i for i in range(5)]
        for i in range(3):
            self.threshold_state.extend(substate)  
    
        # action為acc 的threshold : [0.01~0.5, 0.51~0.99]
        action = []
        b = np.arange(0.01, 0.51, 0.05).tolist()
        g = np.arange(0.50, 1, 0.05).tolist()
        b = [round(num, 2) for num in b]
        g = [round(num, 2) for num in g]
        for i in range(0, 10):
            for j in range(0, 10):
                action.append([b[i], g[j]])
        self.threshold_action = action

        self.threshold_agent = thresholdAgent(self.threshold_buffer)
        print(len(self.threshold_state), len(self.threshold_action),len(self.threshold_state), len(self.threshold_action))
        self.threshold_net = DQN(len(self.threshold_state), len(self.threshold_action)).to(f.device)
        self.threshold_target_net = DQN(len(self.threshold_state), len(self.threshold_action)).to(f.device)
        
        self.threshold_epsilon = r.EPSILON_START    
        self.threshold_optimizer = optim.Adam(self.threshold_net.parameters(), lr=r.LEARNING_RATE)


    def for_slicing_dqn(self):
        self.slicing_buffer = ExperienceBuffer(r.REPLAY_SIZE)

        # state的定義是 [中間 group 各 label 平均 acc, 中間 group  client 數量, 中間 group user 數量]  
        substate = [i for i in range(12)] 
        self.slicing_state= substate
    
        # action為client of intermediate group 
        # self.slicing_action = [i for i in range(1, 101)]
        self.slicing_action = [i for i in range(1, 51)]

        self.slicing_agent = sliceAgent(self.slicing_buffer)

        # dqn model
        print(len(self.slicing_state), len(self.slicing_action),len(self.slicing_state), len(self.slicing_action))
        self.slicing_net = DQN(len(self.slicing_state), len(self.slicing_action)).to(f.device)
        self.slicing_target_net = DQN(len(self.slicing_state), len(self.slicing_action)).to(f.device)
        
        self.slicing_epsilon = r.EPSILON_START    
        self.slicing_optimizer = optim.Adam(self.slicing_net.parameters(), lr=r.LEARNING_RATE)
    

    def state_update_threshold_dqn(self, epoch, groups): 

        # state的定義是 [FL_epoch, 三個 group 人數, intermediate group subset 數, [good group 各 label accuracy], [intermediate group 各 label accuracy], [bad group 各 label accuracy]]
        if len(groups.acc_per_label_good) != 0:
            good = groups.acc_per_label_good
        else:
            good = [0.0] * 10
        if len(groups.acc_per_label_bad) != 0:
            bad = groups.acc_per_label_bad
        else:
            bad = [0.0] * 10
        if len(groups.acc_per_label_intermediate) != 0:
            intermediate = groups.acc_per_label_intermediate
        else:
            intermediate = [0.0] * 10
        state=[epoch, len(groups.good), len(groups.intermediate), len(groups.bad), self.slicing_agent.action]
        state.extend(good)
        state.extend(intermediate)
        state.extend(bad)
        self.threshold_agent.state = state
        
    def state_update_slicing_dqn(self, groups, clients):
    
            # state的定義是 [中間 group 各 label 平均 acc, 中間 group  client 數量, 中間 group user 數量] 
        if len(groups.acc_rec_intermediate) != 0:
            self.slicing_agent.state = groups.acc_per_label_intermediate
        else:
            self.slicing_agent.state = [0.0]*10
        intermediate_users = 0
        for c in groups.intermediate:
            intermediate_users += len(clients[c].local_users)
        self.slicing_agent.state.extend([self.slicing_agent.action, intermediate_users])

        #for client in groups.good:
        #    state = [groups.acc_avg_good, groups.acc_avg_bad, groups.acc_worst_good, groups.acc_worst_bad, 0]  
        #    state.extend(clients[client].acc_per_label)
        #    clients[client].state = state

        #for client in groups.bad:
        #    state = [groups.acc_avg_good, groups.acc_avg_bad, groups.acc_worst_good, groups.acc_worst_bad, 1]  
        #    state.extend(clients[client].acc_per_label)
        #    clients[client].state = state
