from config import for_FL as f
from .Update import Local_process, LocalUpdate_poison

class Attackers():
    
    def __init__(self):
        # 記錄攻擊者的編號
        self.all_attacker = []
        # 攻擊者的總數                                     
        self.attacker_num  = int(f.attack_ratio * f.total_users)
        # 記錄現在有多少攻擊者了    
        self.attacker_count = 0
        # 成為攻擊者的機率                                     
        self.attack_or_not = 1                                      

    
    def poison_setting(self):
        
        # 設定被攻擊的label種類
        # 多一成，多一種被攻擊的label(因為圖片數這樣才夠多)
        if(f.attack_ratio <= 0.1):
            f.target_label = [7]
        elif(f.attack_ratio <= 0.2):
            f.target_label = [7,3]
        elif(f.attack_ratio <= 0.3):
            f.target_label = [7,3,5]
        elif(f.attack_ratio <= 0.4):
            f.target_label = [7,3,5,1]
        elif(f.attack_ratio <= 0.5):
            f.target_label = [7,3,5,1,9]

        print('target_label:',f.target_label)
        print("")

    def reset(self):
        self.all_attacker = []
        # 攻擊者的總數                                     
        self.attacker_num  = int(f.attack_ratio * f.total_users)
        # 記錄現在有多少攻擊者了    
        self.attacker_count = 0
        # 成為攻擊者的機率                                     
        self.attack_or_not = 1  

    def choose_attackers(self, idxs_users, data):
        
        # 決定哪些user是attacker(但還未竄改label)
        for idx in idxs_users:
            # 如果在sampling.py中有改成idxs_labels_sorted，那下面就能直接寫idxs=data.dict_users[idx]
            # 也就是取該user分到的圖片id
            # 否則dict_users的內容其實是{user id: 其分配到的idxs_labels的index}而不是圖片id，很不直觀(我也忘了當初怎麼會這樣寫)
            # 沒改的話就得向下面這樣再從idxs_labels[0]取得真正的圖片id
            initialized = Local_process(dataset=data.dataset_train, idxs=data.idxs_labels[0][data.dict_users[idx]], user_idx=idx, attack_setting=self)
            attack_flag = initialized.split_poison_attackers()
            if(attack_flag):
                self.all_attacker.append(idx)
                self.attacker_count += 1

                # 結果註解掉了，所以不搞機率，先遇到的user就直接定為attacker，直到attacker夠多
                #if(self.attack_or_not>0):
                    #self.attack_or_not -= 1 / (f.total_users*f.attack_ratio)
            
            # 攻擊者已經夠了的話，就停止
            if(self.attacker_count==self.attacker_num):
                break

    
