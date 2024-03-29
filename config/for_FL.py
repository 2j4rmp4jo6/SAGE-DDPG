
#rounds of training
epochs = 5000

#number of clients
num_clients = 10

#number of users per clients
num_users = 5 #100

#number of total users
total_users = 50 #1000

#ratio of attacker in total users
attack_ratio = 0.4

#type of attack
attack_mode = 'poison'

#the poisoned label
target_label = [7,3]

#type of aggregation method
aggregation = 'FedAvg'          

#type of dataset
dataset = 'mnist'               

#GPU ID, -1 for CPU
gpu = 0

# gpu or cpu
device = None

#random seed
# seed = 35                       

#the number of local epochs
local_ep = 5

#local batch size
local_bs = 10

#test batch size
test_bs = 128

#learning rate
lr = 0.01

#SGD momentum
momentum = 0.5

#type of defensive method
defence = 'shuffle'

#the path to save trained model
#要是已經存在的資料夾
model_path = './LessAction/save_model' 

#noniid rate
noniid = 0.4

#scale the user model or not
scale = False

# the label which was attacked random changes to another label or not
target_random = False

# print the training loss in local update or not
local_verbose = False

# 最大 client 數
max_client = 20