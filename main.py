#!/usr/bin/env python3 

import stringprep
# from cv2 import threshold
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import pickle

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from FL_env import FL_env
from config import for_FL as f

# gym.undo_logger_setup()

def train(num_iterations, agent, env,  evaluate, validate_steps, output, restart, max_episode_length=21, debug=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    # bad group threshold 用的 epsilon
    threshold_epsilon = 4
    # 儲存 accuracy 的東西
    # 原本的
    acc_avg_good_n = []
    acc_worst_good_n = []
    # good + intermediate
    acc_avg_good_c = []
    acc_worst_good_c = []
    if restart == 1:
        path_log_accuracy = f.model_path + '_log_accuracy.txt'
        with open(path_log_accuracy, "rb") as file:
            acc_avg_good_n = pickle.load(file)
            acc_worst_good_n = pickle.load(file)
            acc_avg_good_c = pickle.load(file)
            acc_worst_good_c = pickle.load(file)
        print("load sucess!!")
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            print("Episode:", episode)
            observation = deepcopy(env.reset())
            agent.reset(observation)
            last_slicing = 10
            #開始之前先 train 幾次
            for i in range(1):
                print("observation: ", observation)
                observation, reward, done = env.step(episode_steps, np.array([0. ,1., 5]), agent, 1, last_slicing)
        else:
            last_slicing = action[2]
        print("episode_steps: ", episode_steps)
        # agent pick action ...
        if step <= args.warmup and restart == 0:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        # 保護機制拿掉
        # if episode_steps < threshold_epsilon:
        #     action[0] = 0.
        # 這個 decay 可以再調調看
        threshold_epsilon -= 0.002
        print("observation: ", observation)
        print("action: ", action)
        # env response with next_observation, reward, terminate_info
        observation2, reward, done = env.step(episode_steps, action, agent, 0, last_slicing)
        print("reward: ", reward)

        observation2 = env.get_observation(episode_steps, action)
        
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        # if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
        #     policy = lambda x: agent.select_action(x, decay_epsilon=False)
        #     validate_reward = evaluate(env, policy, episode_steps, debug=False, visualize=False)
        #     if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        # if step % int(num_iterations/3) == 0:
        if step % 5 == 0:
            print("save model!")
            agent.save_model(output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )
            # *****之後改*****
            '''
            # 執行一次確定分類效果
            print("--------------------Test good group-------------------------")
            env.final_test_normal(episode_steps)
            # 儲存合併前的結果
            if(len(env.my_clients[0].local_users) > 0):
                acc_good = env.my_clients[0].acc_per_label
            else:
                acc_good = [0.0]*10
            acc_avg_good = np.average(acc_good)
            acc_worst_good = np.min(acc_good)
            acc_avg_good_n.append(acc_avg_good)
            acc_worst_good_n.append(acc_worst_good)

            # 測試 good + intermediate group
            print("--------------------Test good + intermediate group-------------------------")
            env.final_test_combine(episode_steps)
            print("final good accuracy: ", acc_good)
            print("final good accuracy avg: ", acc_avg_good)
            print("final good accuracy min: ", acc_worst_good)
            if(len(env.my_clients[0].local_users) > 0):
                acc_good = env.my_clients[0].acc_per_label
            else:
                acc_good = [0.0]*10
            acc_avg_good = np.average(acc_good)
            acc_worst_good = np.min(acc_good)
            print("final good + intermediate accuracy: ", acc_good)
            print("final good + intermediate accuracy avg: ", acc_avg_good)
            print("final good + intermediate accuracy min: ", acc_worst_good)
            if(len(env.my_clients[1].local_users) > 0):
                acc_bad = env.my_clients[1].acc_per_label
            else:
                acc_bad = [0.0]*10
            acc_avg_bad = np.average(acc_bad)
            acc_worst_bad = np.min(acc_bad)
            print("final bad accuracy: ", acc_bad)
            print("final bad accuracy avg: ", acc_avg_bad)
            print("final bad accuracy min: ", acc_worst_bad)
            # 儲存合併後的結果
            acc_avg_good_c.append(acc_avg_good)
            acc_worst_good_c.append(acc_worst_good)

            print("--------------------End episode-------------------------")
            '''

            # 紀錄 accuracy 走向
            path_log_accuracy = f.model_path + '_log_accuracy.txt'
            with open(path_log_accuracy, "wb") as file:
                pickle.dump(acc_avg_good_n, file)
                pickle.dump(acc_worst_good_n, file)
                pickle.dump(acc_avg_good_c, file)
                pickle.dump(acc_worst_good_c, file)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=200, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO
    parser.add_argument('--restart', default=0, type=int, help='')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    env = FL_env()

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    if args.debug: prPurple("action_space:{}".format(nb_actions))
    if args.debug: prPurple("observation_space:{}".format(nb_states))

    agent = DDPG(nb_states, nb_actions, args)
    evaluate =Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.restart == 1:
        agent.load_weights(args.output)
        env.load(agent)
        print("load success!")

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, args.restart, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
