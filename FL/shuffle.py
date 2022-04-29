from multiprocessing.connection import Client
from ..config import for_FL as f
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
        # self.shuffle_in_intermediate = []
        # for client in clients:
        #     if client.id != 0 and client.id != 1:
        #         self.shuffle_in_intermediate.append(client.id)

        # 對相同group內的users進行suffle
        if(round+1<f.epochs):
            print('shuffle_clients (in intermediate):', self.shuffle_in_intermediate)
            shuffle_user_in_intermediate = []
            # 每個client裡面的人數
            intermediate_client_num = 0
            # client的數量
            client_num = 0
            # 除不盡的user數量
            extra_user_num = 0
            if(len(self.shuffle_in_intermediate)>=1):
                print('Shuffle in intermediate !')
                for id in self.shuffle_in_intermediate:
                    shuffle_user_in_intermediate.extend(clients[id].local_users)    

                # 計算每個client裡面的user數量
                if(len(shuffle_user_in_intermediate) > slice_num):
                    if(len(shuffle_user_in_intermediate) % slice_num == 0):
                        client_num = slice_num
                        intermediate_client_num = len(shuffle_user_in_intermediate) // slice_num
                    else:
                        client_num = slice_num + 1
                        intermediate_client_num = len(shuffle_user_in_intermediate) // slice_num
                        extra_user_num = len(shuffle_user_in_intermediate) - (intermediate_client_num * slice_num)
                else:
                    client_num = len(shuffle_user_in_intermediate)
                    intermediate_client_num = 1
                print('intermediate client number: ', client_num)

                # 找出good, intermediate group最好的client model
                best_client = 0
                max_min_label = 0
                for i in range(len(clients)):
                    if i != 1 and clients[i].acc_per_label_min > max_min_label:
                        best_client = i
                # 之後所有client會重製，所以先備份最好的model
                best_net = copy.deepcopy(clients[best_client].client_net)
                # 重製所有client
                del clients[2 : len(clients) + 2]
                add_client = [Clients(copy.deepcopy(best_net)) for _ in range(client_num)]
                clients.extend(add_client)
                # 分user，除不盡的user給最後一個client
                if(extra_user_num != 0):
                    for i in range(2, client_num + 1):
                        clients[i].local_users = set(np.random.choice(shuffle_user_in_intermediate, intermediate_client_num, replace=False))
                        shuffle_user_in_intermediate = list(set(shuffle_user_in_intermediate) - clients[i].local_users)
                    groups.intermediate = [i for i in range(2,client_num + 1)]
                    clients[client_num + 1].local_users = set(np.random.choice(shuffle_user_in_intermediate, extra_user_num, replace=False))
                    groups.intermediate.append(client_num + 1)
                else:
                    for i in range(2, client_num + 2):
                        clients[i].local_users = set(np.random.choice(shuffle_user_in_intermediate, intermediate_client_num, replace=False))
                        shuffle_user_in_intermediate = list(set(shuffle_user_in_intermediate) - clients[i].local_users)
                    groups.intermediate = [i for i in range(2,client_num + 2)]
        return clients
    
    def execution_group(self, clients, groups, round, threshold):
        for client in clients:
            if client.id != 0 and client.id != 1:
                if client.acc_per_label_min > threshold[1]:
                    self.shuffle_in_good.append(client.id)
                elif client.acc_per_label_min < threshold[0]:
                    self.shuffle_in_bad.append(client.id)
                else:
                    self.shuffle_in_intermediate.append(client.id)

        groups.good = self.shuffle_in_good  
        groups.intermediate = self.shuffle_in_intermediate
        groups.bad = self.shuffle_in_bad

        # 對相同group內的users進行suffle
        if(round+1<f.epochs):
            # good group跟bad group的人數因為不會再變動，因此把這兩個group的user分別放在clint[0], client[1]
            if(len(self.shuffle_in_good)>=1):
                print('Add good user !')
                for id in self.shuffle_in_good:
                    clients[0].local_users.extend(clients[id].local_users)
            if(len(self.shuffle_in_bad)>=1):
                print('Add bad user !')
                for id in self.shuffle_in_bad:
                    clients[1].local_users.extend(clients[id].local_users)

            # shuffle intermediate group
            print('clients (in good):', self.shuffle_in_good)
            print('clients (in intermediate):', self.shuffle_in_intermediate)
            print('clients (in bad):', self.shuffle_in_bad)