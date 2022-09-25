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
        # 對相同 group 內的 users 進行 suffle
        if(round + 1 < f.epochs):
            # 找出最好的 client model
            best_client = 0
            max_min_label = 0
            for i in range(len(clients)):
                if clients[i].acc_per_label_min > max_min_label:
                    best_client = i
            # 之後所有 client 會重製，所以先備份最好的 model
            best_net = copy.deepcopy(clients[best_client].client_net)

            # 先記錄各 group 要做 shuffle 的 user
            shuffle_user_in_good = []
            shuffle_user_in_intermediate = []
            shuffle_user_in_bad = []
            # 各 group client 的數量
            client_num_good = 0
            client_num_intermediate = 0
            client_num_bad = 0
            # 各 group 除不盡的 user 數量
            extra_user_num_good = 0
            extra_user_num_intermediate = 0
            extra_user_num_bad = 0

            # 記錄各 group 的 user 跟 client 數量
            if(len(self.shuffle_in_good) >= 1):
                print('Shuffle in good !')
                for id in self.shuffle_in_good:
                    shuffle_user_in_good.extend(clients[id].local_users)
                
                # 計算要分成幾個 client
                if(len(shuffle_user_in_good) > slice_num):
                    if(len(shuffle_user_in_good) % slice_num == 0):
                        client_num_good = len(shuffle_user_in_good) / slice_num
                    else:
                        client_num_good = len(shuffle_user_in_good) // slice_num + 1
                        extra_user_num_good = len(shuffle_user_in_good) - ((client_num_good - 1) * slice_num)
                # 人數不夠的話就分成一個
                else:
                    client_num_good = 1
            print('good client number: ', client_num_good)

            if(len(self.shuffle_in_intermediate) >= 1):
                print('Shuffle in intermediate !')
                for id in self.shuffle_in_intermediate:
                    shuffle_user_in_intermediate.extend(clients[id].local_users)
                
                if(len(shuffle_user_in_intermediate) > slice_num):
                    if(len(shuffle_user_in_intermediate) % slice_num == 0):
                        client_num_intermediate = len(shuffle_user_in_intermediate) / slice_num
                    else:
                        client_num_intermediate = len(shuffle_user_in_intermediate) // slice_num + 1
                        extra_user_num_intermediate = len(shuffle_user_in_intermediate) - ((client_num_intermediate - 1) * slice_num)
                else:
                    client_num_intermediate = 1
            print('intermediate client number: ', client_num_intermediate)

            if(len(self.shuffle_in_bad) >= 1):
                print('Shuffle in bad !')
                for id in self.shuffle_in_bad:
                    shuffle_user_in_bad.extend(clients[id].local_users)

                if(len(shuffle_user_in_bad) > slice_num):
                    if(len(shuffle_user_in_bad) % slice_num == 0):
                        client_num_bad = len(shuffle_user_in_bad) / slice_num
                    else:
                        client_num_bad = len(shuffle_user_in_bad) // slice_num + 1
                        extra_user_num_bad = len(shuffle_user_in_bad) - ((client_num_bad - 1) * slice_num)
                else:
                    client_num_bad = 1
            print('bad client number: ', client_num_bad)

            # 重製所有 client
            clients = [Clients(copy.deepcopy(best_net)) for _ in range(int(client_num_good + client_num_intermediate + client_num_bad))]

            # 前幾個 client 給 good group
            good_id_end = int(client_num_good)
            groups.good = [i for i in range(good_id_end)]
            # 之後給 intermediate group
            intermediate_id_end = int(client_num_good + client_num_intermediate)
            groups.intermediate = [i for i in range(good_id_end, intermediate_id_end)]
            # 剩下給 bad group
            bad_id_end = int(client_num_good + client_num_intermediate + client_num_bad)
            groups.bad = [i for i in range(intermediate_id_end, bad_id_end)]

            # 開始分 user
            if(len(self.shuffle_in_good) >= 1):
                if(extra_user_num_good != 0):
                    for id in range(good_id_end - 1):
                        clients[id].local_users = set(np.random.choice(shuffle_user_in_good, slice_num, replace=False))
                        shuffle_user_in_good = list(set(shuffle_user_in_good) - clients[id].local_users)
                    clients[good_id_end - 1].local_users = set(np.random.choice(shuffle_user_in_good, extra_user_num_good, replace=False))
                elif(client_num_good == 1):
                    clients[0].local_users = set(shuffle_user_in_good)
                else:
                    for id in range(good_id_end):
                        clients[id].local_users = set(np.random.choice(shuffle_user_in_good, slice_num, replace=False))
                        shuffle_user_in_good = list(set(shuffle_user_in_good) - clients[id].local_users)

            if(len(self.shuffle_in_intermediate) >= 1):
                if(extra_user_num_intermediate != 0):
                    for id in range(good_id_end, intermediate_id_end - 1):
                        clients[id].local_users = set(np.random.choice(shuffle_user_in_intermediate, slice_num, replace=False))
                        shuffle_user_in_intermediate = list(set(shuffle_user_in_intermediate) - clients[id].local_users)
                    clients[intermediate_id_end - 1].local_users = set(np.random.choice(shuffle_user_in_intermediate, extra_user_num_intermediate, replace=False))
                elif(client_num_intermediate == 1):
                    clients[good_id_end].local_users = set(shuffle_user_in_intermediate)
                else:
                    for id in range(good_id_end, intermediate_id_end):
                        clients[id].local_users = set(np.random.choice(shuffle_user_in_intermediate, slice_num, replace=False))
                        shuffle_user_in_intermediate = list(set(shuffle_user_in_intermediate) - clients[id].local_users)

            if(len(self.shuffle_in_bad) >= 1):
                if(extra_user_num_bad != 0):
                    for id in range(intermediate_id_end, bad_id_end - 1):
                        clients[id].local_users = set(np.random.choice(shuffle_user_in_bad, slice_num, replace=False))
                        shuffle_user_in_bad = list(set(shuffle_user_in_bad) - clients[id].local_users)
                    clients[bad_id_end - 1].local_users = set(np.random.choice(shuffle_user_in_bad, extra_user_num_bad, replace=False))
                elif(client_num_bad == 1):
                    clients[intermediate_id_end].local_users = set(shuffle_user_in_bad)
                else:
                    for id in range(intermediate_id_end, bad_id_end):
                        clients[id].local_users = set(np.random.choice(shuffle_user_in_bad, slice_num, replace=False))
                        shuffle_user_in_bad = list(set(shuffle_user_in_bad) - clients[id].local_users)
                        
        return clients
    
    def execution_group(self, clients, groups, round, threshold_bad, threshold_good, pre_train):
        '''
        for client in clients:
            if client.id != 0 and client.id != 1:
                if client.acc_per_label_min >= threshold_good and pre_train != 1:
                    self.shuffle_in_good.append(client.id)
                elif client.acc_per_label_min <= threshold_bad and pre_train != 1:
                    self.shuffle_in_bad.append(client.id)
                else:
                    self.shuffle_in_intermediate.append(client.id)
        '''

        # 讓 good, bad 的 client 也可以 shuffle
        for client in clients:
            if client.acc_per_label_min >= threshold_good and pre_train != 1:
                self.shuffle_in_good.append(client.id)
            elif client.acc_per_label_min <= threshold_bad and pre_train != 1:
                self.shuffle_in_bad.append(client.id)
            else:
                self.shuffle_in_intermediate.append(client.id)

        groups.good = self.shuffle_in_good  
        groups.intermediate = self.shuffle_in_intermediate
        groups.bad = self.shuffle_in_bad

        # shuffle 會在 execution_client() 進行，這邊先拿掉
        '''
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
        '''

        print('clients (in good):', self.shuffle_in_good)
        print('clients (in intermediate):', self.shuffle_in_intermediate)
        print('clients (in bad):', self.shuffle_in_bad)