from cProfile import label
from config import for_FL as f
import matplotlib.pyplot as plt
import pickle

path_log_variable = f.model_path + '_log_variable.txt'
with open(path_log_variable, "rb") as file:
    total_rewards = pickle.load(file)
    value_loss_record = pickle.load(file)
    policy_loss_record = pickle.load(file)
    normal_ratio_good = pickle.load(file)
    normal_ratio_bad = pickle.load(file)
    attacker_ratio_bad = pickle.load(file)
    attacker_ratio_good = pickle.load(file)

'''
path_log_accuracy = f.model_path + '_log_accuracy.txt'
with open(path_log_accuracy, "rb") as file:
    acc_avg_good_n = pickle.load(file)
    acc_worst_good_n = pickle.load(file)
    acc_avg_good_c = pickle.load(file)
    acc_worst_good_c = pickle.load(file)
'''

episode = range(len(total_rewards))
print('Episode: ', len(total_rewards))
plt.plot(episode, total_rewards, color=(255/255, 0/255, 0/255))
plt.title("Total Reward")
plt.ylabel("Total Reward")
plt.xlabel("Episode")
plt.grid(True)
plt.savefig(f.model_path[0:-10]+'total_rewards.jpg')
plt.show()
plt.close()

episode = range(len(value_loss_record))
plt.plot(episode, value_loss_record, color=(0/255, 0/255, 255/255))
plt.title("Value loss")
plt.ylabel("Loss")
plt.xlabel("Episode")
plt.grid(True)
plt.savefig(f.model_path[0:-10]+'value_loss_record.jpg')
plt.show()
plt.close()

plt.plot(episode, policy_loss_record, color=(0/255, 0/255, 255/255))
plt.title("Policy loss")
plt.ylabel("Loss")
plt.xlabel("Episode")
plt.grid(True)
plt.savefig(f.model_path[0:-10]+'policy_loss_record.jpg')
plt.show()
plt.close()

episode = range(len(normal_ratio_good))
plt.plot(episode, normal_ratio_good, color=(0/255, 255/255, 0/255))
plt.title("Normal user true positive ratio")
plt.ylabel("Normal user ratio")
plt.xlabel("Episode")
plt.grid(True)
plt.savefig(f.model_path[0:-10]+'normal_ratio_good.jpg')
plt.show()
plt.close()

plt.plot(episode, normal_ratio_bad, color=(0/255, 255/255, 0/255))
plt.title("Normal user false positive ratio")
plt.ylabel("Normal user ratio")
plt.xlabel("Episode")
plt.grid(True)
plt.savefig(f.model_path[0:-10]+'normal_ratio_bad.jpg')
plt.show()
plt.close()

plt.plot(episode, attacker_ratio_bad, color=(0/255, 255/255, 0/255))
plt.title("Attacker true negative ratio")
plt.ylabel("Attacker ratio")
plt.xlabel("Episode")
plt.grid(True)
plt.savefig(f.model_path[0:-10]+'attacker_ratio_bad.jpg')
plt.show()
plt.close()

plt.plot(episode, attacker_ratio_good, color=(0/255, 255/255, 0/255))
plt.title("Attacker false negative ratio")
plt.ylabel("Attacker ratio")
plt.xlabel("Episode")
plt.grid(True)
plt.savefig(f.model_path[0:-10]+'attacker_ratio_good.jpg')
plt.show()
plt.close()

'''
episode = range(len(acc_avg_good_n))
plt.plot(episode, acc_avg_good_n, color=(0/255, 255/255, 255/255), label="good")
plt.plot(episode, acc_avg_good_c, color=(128/255, 138/255, 135/255), label="good + intermsdiate")
plt.title("Good group average accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Episode")
plt.grid(True)
plt.legend()
plt.savefig(f.model_path[0:-10]+'acc_avg_good_n.jpg')
plt.show()
plt.close()

plt.plot(episode, acc_worst_good_n, color=(0/255, 255/255, 255/255), label="good")
plt.plot(episode, acc_worst_good_c, color=(128/255, 138/255, 135/255), label="good + intermsdiate")
plt.title("Good group worst accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Episode")
plt.grid(True)
plt.legend()
plt.savefig(f.model_path[0:-10]+'acc_worst_good_n.jpg')
plt.show()
plt.close()
'''