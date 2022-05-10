from config import for_FL as f
import matplotlib.pyplot as plt
import pickle

path_log_variable = f.model_path + '_log_variable.txt'
with open(path_log_variable, "rb") as file:
    total_rewards = pickle.load(file)
    value_loss_record = pickle.load(file)
    policy_loss_record = pickle.load(file)
    attacker_ratio_bad = pickle.load(file)
    attacker_ratio_good = pickle.load(file)

episode = range(len(total_rewards))
print('Episode: ', len(total_rewards))
plt.plot(episode, total_rewards, color=(255/255, 0/255, 0/255))
plt.title("Total Reward")
plt.ylabel("Total Reward")
plt.xlabel("Episode")
plt.grid(True)
plt.show()

episode = range(len(value_loss_record))
plt.plot(episode, value_loss_record, color=(0/255, 0/255, 255/255))
plt.title("Value loss")
plt.ylabel("Loss")
plt.xlabel("Episode")
plt.grid(True)
plt.show()

plt.plot(episode, policy_loss_record, color=(0/255, 0/255, 255/255))
plt.title("Policy loss")
plt.ylabel("Loss")
plt.xlabel("Episode")
plt.grid(True)
plt.show()

episode = range(len(attacker_ratio_bad))
plt.plot(episode, attacker_ratio_bad, color=(0/255, 255/255, 0/255))
plt.title("Attacker true positive ratio")
plt.ylabel("Attacker ratio")
plt.xlabel("Episode")
plt.grid(True)
plt.show()