from config import for_FL as f
import numpy as np
import copy
from .Update import LocalUpdate_poison
from .Fed import FedAvg
from .test import test_img_poison
from datetime import datetime
import time

import torch.multiprocessing as mp

#要在 process 間共用 GPU 資料的話要用這個模式開 multiprocess
ctx = mp.get_context("spawn")

#有要用 lock 的話要在 初始化 pool 的時候跑這個(但現在沒有用到
def init_pool_processes(the_lock):
    '''Initialize each process with a global variable lock.
    '''
    global lock
    lock = the_lock

# 排序 process 回傳結果用的東西
def sort_key(output):
    return output[0]

# np.random.seed(f.seed)

class Client():
    ID = 0
    def __init__(self,net):
        self.id = Client.ID
        #該client分到的users
        self.local_users = []
        #該client的model net，所有client的初始net都相同          
        self.client_net = net
        #該client分到的攻擊者的編號           
        self.attacker_idxs = []
        #屬於哪個group         
        self.belong_to = 0              
        self.weights = []
        self.loss = []
        # 好像沒用到這兩變數
        self.training_acc = 0
        self.testing_acc = 0
        # user_sizes是每個user有的照片量
        self.user_sizes = None          

        # 各user回傳的training loss的平均
        self.loss_avg = 0     
        # 雖然寫test，但其實是validation結果          
        self.acc_test = 0
        self.loss_test = 0
        self.acc_per_label = None
        self.poison_acc = 0
        self.acc_per_label_avg = 0
        # 各label acc最小值
        self.acc_per_label_min = 0
        self.user_loss = {}

        self.state = None
        self.old_state = None
        self.action = None

        Client.ID += 1

    def reset(self,epoch):
        if epoch==1:
            self.attacker_idxs = []
        self.weights = []
        self.user_loss = {}
        self.loss = []

    def split_user_to_client(self,all_users,attackers):
        
        # 隨機選
        self.local_users = set(np.random.choice(all_users, f.num_users, replace=False))
        # 選過的不再選
        all_users = list(set(all_users) - self.local_users)
        
        # 若有選到攻擊者，則記錄
        for i in self.local_users:
            if i in attackers and i not in self.attacker_idxs:
                self.attacker_idxs.append(i)

        return all_users

    # 每個 process 要做的事 (user local train)
    def process(self, local, net, idx):
        # 這裡的deepcopy是因為master model分給其user,這些model是在各user是獨立訓練的
        # w, loss, attack_flag = local.train(net=copy.deepcopy(self.client_net).to(f.device))
        w, loss, attack_flag = local.train(net)
        
        # print("finish idx: ", idx)
        # 把結果打包好傳回去
        return (idx, copy.deepcopy(w), copy.deepcopy(loss), attack_flag)

    def local_update_poison(self,data,all_attacker,round):
        
        # 等等存 process 所需資料的 list
        args = []
        # 初始化一個 pool，processes 為這個 pool 裡的 process 個數
        pool = ctx.Pool(processes = 4)
        # pool = ctx.Pool(processes = 2, initializer = init_pool_processes, initargs=(lock,))

        # torch.multiprocessing.spawn(process, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')

        net=copy.deepcopy(self.client_net).to(f.device)
        
        #讓不同的 process 能使用同一個 net 進行訓練
        net.share_memory()

        # local user 本來沒有照順序，但為了後面等各 process 的資料及 weight 對應方便加上排序
        print(self.local_users)
        users = list(self.local_users)
        users.sort()

        self.local_users = users

        for idx in self.local_users:
            # 這邊的idxs的值，看先前有沒有改過idxs_labels_sorted，有的話記得也要改
            # 這邊會竄改label(如果是攻擊者的話)，並且各user會訓練model (local training)
            local = LocalUpdate_poison(dataset=data.dataset_train, idxs=data.idxs_labels[0][data.dict_users[idx]], user_idx=idx, attack_idxs=all_attacker)
            
            # 準備各process需要的資料
            args.append((local, net, idx, ))
            
            ################# 原本的 ################
            # # 這裡的deepcopy是因為master model分給其user,這些model是在各user是獨立訓練的
            # # w, loss, attack_flag = local.train(net=copy.deepcopy(self.client_net).to(f.device))
            # w, loss, attack_flag = local.train(net)
            
            # # 紀錄攻擊者
            # # round 0在上面的split_user_to_client會計算一次了，所以這邊不算
            # if(attack_flag and idx not in self.attacker_idxs):
            #     self.attacker_idxs.append(idx)
            
            # self.weights.append(copy.deepcopy(w))
            # self.loss.append(copy.deepcopy(loss))
            # self.user_loss[idx] = copy.deepcopy(loss)
            ################# 原本的 ################

        # 將準備好的資料分給各 process
        pool_outputs = pool.starmap(self.process, args, )

        # 關閉pool並等待所有 process 結束
        pool.close()
        pool.join()
        
        # 把收到的結果案到 user idx 排序，然後存進對應的變數
        pool_outputs.sort(key = sort_key)
        for idx in range(len(pool_outputs)):
            # print(idx)
            if(pool_outputs[idx][3] and idx not in self.attacker_idxs):
                    self.attacker_idxs.append(idx)
            self.weights.append(pool_outputs[idx][1])
            self.loss.append(pool_outputs[idx][2])
            self.user_loss[idx] = pool_outputs[idx][2]
        
        print("Client {}".format(self.id))
        print(" {}/{} are attackers with {} attack ".format(len(self.attacker_idxs), len(self.local_users), f.attack_mode))

        # 根據照片數的權重，來進行各user的模型參數的平均
        self.user_sizes = np.array([ len(data.dict_users[idx]) for idx in self.local_users ])
        user_weights = self.user_sizes / float(sum(self.user_sizes))
        
        # aggregation的方法用最普通的FedAvg
        if f.aggregation == "FedAvg":
            w_glob = FedAvg(self.weights, user_weights)
        else:
            print('no other aggregation method.')
            exit()

        # 新的master model
        self.client_net.load_state_dict(w_glob)
        self.loss_avg = np.sum(self.loss * user_weights)
        
        print('=== Round {:3d}, Average loss {:.6f} ==='.format(round, self.loss_avg))
        print(" {} users; time {}".format(len(self.local_users), datetime.now().strftime("%H:%M:%S")) )

    def show_testing_result(self,my_data):
            start_time = time.time()
            
            # 進行validation
            self.acc_test, self.loss_test, self.acc_per_label, self.poison_acc = test_img_poison(self.client_net.to(f.device), my_data.dataset_test)
            self.acc_per_label_avg = sum(self.acc_per_label)/len(self.acc_per_label)
            self.acc_per_label_min = min(self.acc_per_label)
            
            print( " Testing accuracy: {} loss: {:.6}".format(self.acc_test, self.loss_test))
            print( " Testing Label Acc: {}".format(self.acc_per_label) )
            print( " Testing Avg Label Acc : {}".format(self.acc_per_label_avg))
            print( " Testing Min Label Acc : {}".format(self.acc_per_label_min))
            if f.attack_mode=='poison':
                print( " Poison Acc: {}".format(self.poison_acc) )
            
            end_time = time.time()
            print('')
            
            return end_time - start_time