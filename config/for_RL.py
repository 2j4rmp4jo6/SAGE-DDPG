
RL_MODEL = 'dqn'

BATCH_SIZE = 40

#buffer size
REPLAY_SIZE = 50  

LEARNING_RATE = 1e-3

#多久才更新一次target_net
SYNC_TARGET_FRAMES = 10

#累積足夠數據才開始訓練dqn
REPLAY_START_SIZE = 50

#epsilon在幾步中慢慢變化
EPSILON_DECAY_LAST_FRAME = 2000
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

# 用於更新net
GAMMA = 0.99