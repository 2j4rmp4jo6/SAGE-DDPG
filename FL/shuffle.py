from multiprocessing.connection import Client
from config import for_FL as f
import numpy as np
from .clients import Client as Clients
import copy

class Shuffle():

    def __init__(self):
        # 紀錄good group 中的 clients
        self.shuffle_in_good = []
        # 紀錄bad group 中的 clients
        self.shuffle_in_bad = []
        # 好像沒用到這兩變數
        self.acc_in_good = []
        self.acc_in_bad = []

        # intermediate group用到的
        self.shuffle_in_intermediate = []

    def reset(self):
        self.shuffle_in_good = []
        self.shuffle_in_bad = []
        self.acc_in_good = []
        self.acc_in_bad = []

        # intermediate group用到的
        self.shuffle_in_intermediate = []

    def execution_client(self, clients, groups, round, slice_num):
        if(round + 1 < f.epochs):
            print('shuffle_clients (in good):', self.shuffle_in_good)
            shuffle_user_in_good = []
            if(len(self.shuffle_in_good) > 1):
                print('Shuffle in good !')
                for id in self.shuffle_in_good:
                    shuffle_user_in_good.extend(clients[id].local_users)
                
                for id in self.shuffle_in_good:
                    clients[id].local_users = set(np.random.choice(shuffle_user_in_good, f.num_users, replace=False))
                    shuffle_user_in_good = list(set(shuffle_user_in_good) - clients[id].local_users)
            
            print('shuffle_clients (in intermediate):', self.shuffle_in_intermediate)
            shuffle_user_in_intermediate = []
            if(len(self.shuffle_in_intermediate) > 1):
                print('Shuffle in intermediate !')
                for id in self.shuffle_in_intermediate:
                    shuffle_user_in_intermediate.extend(clients[id].local_users)
                
                for id in self.shuffle_in_intermediate:
                    clients[id].local_users = set(np.random.choice(shuffle_user_in_intermediate, f.num_users, replace=False))
                    shuffle_user_in_intermediate = list(set(shuffle_user_in_intermediate) - clients[id].local_users)
            
            print('shuffle_clients (in bad):', self.shuffle_in_bad)
            shuffle_user_in_bad = []
            if(len(self.shuffle_in_bad) > 1):
                print('Shuffle in bad !')
                for id in self.shuffle_in_bad:
                    shuffle_user_in_bad.extend(clients[id].local_users)
                
                for id in self.shuffle_in_bad:
                    clients[id].local_users = set(np.random.choice(shuffle_user_in_bad, f.num_users, replace=False))
                    shuffle_user_in_bad = list(set(shuffle_user_in_bad) - clients[id].local_users)
    
    def execution_group(self, clients, groups, round, threshold_bad, threshold_good, pre_train):
        clients_sort = sorted(clients, key = lambda client : client.acc_per_label_min)
        clients_sort.reverse()
        if pre_train != 1:
            for i in range(len(clients_sort)):
                if i < threshold_good:
                    self.shuffle_in_good.append(clients_sort[i].id)
                elif i < threshold_good + threshold_bad:
                    self.shuffle_in_intermediate.append(clients_sort[i].id)
                else:
                    self.shuffle_in_bad.append(clients_sort[i].id)
        else:
            self.shuffle_in_intermediate = [i for i in range(f.num_clients)]

        groups.good = self.shuffle_in_good  
        groups.intermediate = self.shuffle_in_intermediate
        groups.bad = self.shuffle_in_bad

        print('clients (in good):', self.shuffle_in_good)
        print('clients (in intermediate):', self.shuffle_in_intermediate)
        print('clients (in bad):', self.shuffle_in_bad)