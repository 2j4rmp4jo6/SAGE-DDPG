'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/test.py
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import for_FL as f
import numpy as np
import random

f.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')

def test_img_poison(net, datatest):

    net.eval()
    test_loss = 0
    if f.dataset == "mnist":
        # 各種圖預測正確的數量
        correct  = torch.tensor([0.0] * 10)
        # 各種圖的數量
        gold_all = torch.tensor([0.0] * 10)
    else:
        print("Unknown dataset")
        exit(0)

    # 攻擊效果
    poison_correct = 0.0

    data_loader = DataLoader(datatest, batch_size=f.test_bs)
    
    print(' test data_loader(per batch size):',len(data_loader))
    
    '''
    for idx, (data, target) in enumerate(data_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)
        
        log_probs = net(data.float())
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # 預測解
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # 正解
        y_gold = target.data.view_as(y_pred).squeeze(1)
        
        y_pred = y_pred.squeeze(1)


        for pred_idx in range(len(y_pred)):
            
            gold_all[ y_gold[pred_idx] ] += 1
            
            # 預測和正解相同
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1
            elif f.attack_mode == 'poison':
                # 被攻擊的目標，攻擊效果如何
                for label in f.target_label:
                    if int(y_pred[pred_idx]) != label and int(y_gold[pred_idx]) == label:
                        poison_correct += 1
    '''

    # 儲存要一次傳的資料的
    data_test = np.array([])
    target_test = np.array([])
    # 儲存要補進數量不足的資料的
    data_tmp = np.array([])
    target_tmp = np.array([])
    for idx, (data, target) in enumerate(data_loader):
        if f.gpu != -1:
            # 隨機取一張放進 tmp，先用第一章圖定義形狀
            if len(data_tmp) == 0 and data.shape[0] > 1:
                r = random.randint(1, data.shape[0] - 1)
                data_tmp = np.expand_dims(data[r], axis=0)
                target_tmp = target[r]
            elif data.shape[0] > 1:
                r = random.randint(1, data.shape[0] - 1)
                data_tmp = np.append(data_tmp, np.expand_dims(data[r], axis=0), axis=0)
                target_tmp = np.append(target_tmp, target[r])
            # 因為不知道 data 的 size，所以用第一筆決定
            if len(data_test) == 0:
                data_test = np.expand_dims(data, axis=0)
                target_test = np.expand_dims(target, axis=0)
            # 剩下的用 append
            else:
                # 數量不足的補上
                while data.shape[0] < data_test.shape[1]:
                    r = random.randint(1, data_tmp.shape[0] - 1)
                    data = np.append(data, np.expand_dims(data_tmp[r], axis=0), axis=0)
                    target = np.append(target, np.expand_dims(target_tmp[r], axis=0), axis=0)
                # 加進 training 的 array
                data_test = np.append(data_test, np.expand_dims(data, axis=0), axis=0)
                target_test = np.append(target_test, np.expand_dims(target, axis=0), axis=0)
    # 一起傳進 gpu
    data_test, target_test = torch.from_numpy(data_test).to(f.device), torch.from_numpy(target_test).to(f.device)
    # 開始 test
    for i in range(len(data_test)):
        log_probs = net(data_test[i].float())
        test_loss += F.cross_entropy(log_probs, target_test[i], reduction='sum').item()
        # 預測解
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # 正解
        y_gold = target_test[i].data.view_as(y_pred).squeeze(1)
        
        y_pred = y_pred.squeeze(1)


        for pred_idx in range(len(y_pred)):
            
            gold_all[ y_gold[pred_idx] ] += 1
            
            # 預測和正解相同
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1
            elif f.attack_mode == 'poison':
                # 被攻擊的目標，攻擊效果如何
                for label in f.target_label:
                    if int(y_pred[pred_idx]) != label and int(y_gold[pred_idx]) == label:
                        poison_correct += 1

    test_loss /= len(data_loader.dataset)

    accuracy = (sum(correct) / sum(gold_all)).item()
    
    acc_per_label = correct / gold_all

    poison_acc = 0

    if(f.attack_mode == 'poison'):
        tmp = 0
        for label in f.target_label:
            tmp += gold_all[label].item()
        poison_acc = poison_correct/tmp
    
    return accuracy, test_loss, acc_per_label.tolist(), poison_acc





