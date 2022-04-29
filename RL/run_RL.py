from ..config import for_RL as r
from ..config import for_FL as f
from .dqn_setting import calc_threshold_loss, calc_slicing_loss

def run_threshold(rl_start, rl_cnt, fl_epoch, clients, rl, threshold_rl_loss, not_final_rond):
    # 隨機選動作 or 照著RL model所給的動作 的 機率
    if fl_epoch ==21:
        epsilon = 0.0
    else:
        epsilon = max(r.EPSILON_FINAL, r.EPSILON_START - rl.threshold_count / r.EPSILON_DECAY_LAST_FRAME)
    
    # 開始能比較前後，得reward
    if(rl_start==1):
        # for client in range(f.num_clients-1):
        #     reward = rl.agent.get_reward(clients[client].old_state, clients[client].state, False)
        # reward = rl.agent.get_reward(clients[f.num_clients-1].old_state, clients[f.num_clients-1].state, False)
        rl.threshold_agent.get_reward(rl.threshold_agent.old_state, rl.threshold_agent.state, False, clients, rl.threshold_agent.action, rl.threshold_agent.action, not_final_rond)
        
        # print('total_reward:',rl.agent.get_total_reward())
        # rl.agent.reset_total_reward()
        
        if len(rl.threshold_agent.exp_buffer) >= r.REPLAY_START_SIZE:
            # 這邊應該是在更新RL model
            # Update the target network, copying all weights and biases in DQN
            if rl.threshold_count % r.SYNC_TARGET_FRAMES == 0:
                rl.threshold_target_net.load_state_dict(rl.threshold_net.state_dict())
            
            # reset the gradients of model parameters.
            rl.threshold_optimizer.zero_grad()
            batch = rl.threshold_buffer.sample(r.BATCH_SIZE)
            loss_t = calc_threshold_loss(batch, rl.threshold_net, rl.threshold_target_net, f.device )
            threshold_rl_loss.append(loss_t.data.item())
            print('threshold dqn loss:',loss_t)   
  

            # Backpropagate the prediction loss 
            loss_t.backward()

            # adjust the parameters by the gradients collected in the backward pass.
            rl.threshold_optimizer.step()
                           
    
    # 最後一round只算reward的話不執行
    if not_final_rond == 1:
        rl.threshold_agent.choose_action(rl.threshold_agent.state, rl.threshold_net, fl_epoch, epsilon, f.device)
        print('threshold action: ', rl.threshold_agent.action)
        
    rl.threshold_count += 1
    rl.threshold_agent.old_state = rl.threshold_agent.state
    
    rl_cnt += 1
    return rl_start, rl_cnt
    
def run_slicing(rl_start, rl_cnt, fl_epoch, clients, rl, slicing_rl_loss, not_final_rond):
    # 隨機選動作 or 照著RL model所給的動作 的 機率
    if fl_epoch ==21:
        epsilon = 0.0
    else:
        epsilon = max(r.EPSILON_FINAL, r.EPSILON_START - rl.slicing_count / r.EPSILON_DECAY_LAST_FRAME)
        
    if(rl_start==1):
        rl.slicing_agent.get_reward(rl.slicing_agent.old_state, rl.slicing_agent.state, False, clients, rl.threshold_agent.action, rl.slicing_agent.action, not_final_rond)

        if len(rl.slicing_agent.exp_buffer) >= r.REPLAY_START_SIZE:
            # 這邊應該是在更新RL model
            # Update the target network, copying all weights and biases in DQN
            if rl.slicing_count  % r.SYNC_TARGET_FRAMES == 0:
                rl.slicing_target_net.load_state_dict(rl.slicing_net.state_dict())
            
            # reset the gradients of model parameters.
            rl.slicing_optimizer.zero_grad()
            batch = rl.slicing_buffer.sample(r.BATCH_SIZE)
            loss_t_slicing = calc_slicing_loss(batch, rl.slicing_net, rl.slicing_target_net, f.device )
            slicing_rl_loss.append(loss_t_slicing.data.item())
            print('slice dqn loss:',loss_t_slicing)   

            # Backpropagate the prediction loss 
            loss_t_slicing.backward()

            # adjust the parameters by the gradients collected in the backward pass.
            rl.slicing_optimizer.step()                    

    # 最後一round只算reward的話不執行
    if not_final_rond == 1:
        rl.slicing_agent.choose_action(rl.slicing_agent.state, rl.slicing_net, fl_epoch, epsilon, f.device)
        print('slice action:',rl.slicing_agent.action)   

    rl.slicing_count += 1
    rl.slicing_agent.old_state = rl.slicing_agent.state

    rl_cnt += 1
    return rl_start, rl_cnt
