from ..config import for_FL as f
import numpy as np

# np.random.seed(f.seed)

class Groups():

    def __init__(self):
        #good group分到的clients
        self.good = []
        #bad group分到的clients                   
        self.bad = []
        #記錄good group中的各client的acc                                                   
        self.acc_rec_good = []
        #記錄bad group中的各client的acc                                          
        self.acc_rec_bad = []                                           
        self.acc_worst_good = 0
        self.acc_worst_bad = 0
        self.acc_avg_good = 0
        self.acc_avg_bad = 0

        # 算各個group的人數
        self.num_users_good = 0
        self.num_users_bad = 0
        self.num_users_intermediate = 0

        # 算各個group 各 label的 acc
        # 算法是先將各client的各label acc*client的人數，最後除以各group總人數
        self.acc_per_label_good = [0.0] * 10
        self.acc_per_label_bad = [0.0] * 10
        self.acc_per_label_intermediate = [0.0] * 10

        # intermediate group用到的 (最初所有client都在此處)
        self.intermediate = [i for i in range(f.num_clients+2)]
        self.acc_rec_intermediate = []
        self.acc_worst_intermediate = 0
        self.acc_avg_intermediate = 0

    def reset(self, epoch, slice_num ):
        self.acc_rec_good = []
        self.acc_rec_bad = []
        self.acc_worst_good = 0
        self.acc_worst_bad = 0
        self.acc_avg_good = 0
        self.acc_avg_bad = 0

        self.num_users_good = 0
        self.num_users_bad = 0
        self.num_users_intermediate = 0
        
        self.acc_per_label_good = [0.0] * 10
        self.acc_per_label_bad = [0.0] * 10
        self.acc_per_label_intermediate = [0.0] * 10

        # intermediate group用到的
        self.acc_rec_intermediate = []
        self.acc_worst_intermediate = 0
        self.acc_avg_intermediate = 0

        if epoch == 1:
            self.good = []
            self.bad = []
            self.intermediate = [i for i in range(f.num_clients+2)]


    def record(self,client):
        if(client.id in self.good):
            self.acc_rec_good.append(client.acc_per_label_avg)
            self.num_users_good += len(client.local_users)
        
        if(len(self.acc_rec_good)>0):
            self.acc_avg_good = np.average(self.acc_rec_good)
            self.acc_worst_good = np.min(self.acc_rec_good)

            # 先將各client的各label acc*client的人數
            acc_per_label = [i*len(client.local_users) for i in client.acc_per_label]
            print(acc_per_label)
            self.acc_per_label_good = [a+b for a,b in zip(self.acc_per_label_good, acc_per_label)]
            
        if(client.id in self.bad):
            self.acc_rec_bad.append(client.acc_per_label_avg)
            self.num_users_bad += len(client.local_users)
        
        if(len(self.acc_rec_bad)>0):
            self.acc_avg_bad = np.average(self.acc_rec_bad)
            self.acc_worst_bad = np.min(self.acc_rec_bad)
            
            # 先將各client的各label acc*client的人數
            acc_per_label = [i*len(client.local_users) for i in client.acc_per_label]
            self.acc_per_label_bad = [a+b for a,b in zip(self.acc_per_label_bad, acc_per_label)]

        # intermediate group用到的
        if(client.id in self.intermediate):
            self.acc_rec_intermediate.append(client.acc_per_label_avg)
            self.num_users_intermediate += len(client.local_users)
        
        if(len(self.acc_rec_intermediate)>0):
            self.acc_avg_intermediate = np.average(self.acc_rec_intermediate)
            self.acc_worst_intermediate = np.min(self.acc_rec_intermediate)

            # 先將各client的各label acc*client的人數
            acc_per_label = [i*len(client.local_users) for i in client.acc_per_label]
            self.acc_per_label_intermediate = [a+b for a,b in zip(self.acc_per_label_intermediate, acc_per_label)]


        # print("======")
        # print("")
