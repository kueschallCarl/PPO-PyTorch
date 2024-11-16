import os
import glob
from datetime import datetime
import torch
import numpy as np
import gym
from models.ppo import PPO
from utils.wrappers import PettingZooWrapper
from pettingzoo.mpe import simple_v3

def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "simple_v3"
    raw_env = simple_v3.parallel_env()

    has_continuous_action_space = True

    max_ep_len = 1000
    max_training_timesteps = int(3e6)

    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 2
    save_model_freq = int(1e5)

    action_std = 0.6
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = int(2.5e5)

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001
    random_seed = 0
    #####################################################

    print("training environment name : " + env_name)

    # Create env
    raw_env = simple_v3.parallel_env(continuous_actions=True)
    first_agent = raw_env.possible_agents[0]
    
    # Get state and action dimensions
    state_dim = raw_env.observation_space(first_agent).shape[0]
    if has_continuous_action_space:
        action_dim = raw_env.action_space(first_agent).shape[0]
    else:
        action_dim = raw_env.action_space(first_agent).n

    env = PettingZooWrapper(raw_env, num_agents=len(raw_env.possible_agents))

    # Set up logging
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    run_num = len(next(os.walk(log_dir))[2])
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    # Set up model saving
    directory = "PPO_preTrained"
    if not os.path.exists(directory): os.makedirs(directory)
    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory): os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num)

    # Initialize agents
    ppo_agents = [
        PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
            has_continuous_action_space, action_std) 
        for _ in range(len(raw_env.possible_agents))
    ]

    # Logging
    print("Started training at (GMT) : ", datetime.now().replace(microsecond=0))
    start_time = datetime.now().replace(microsecond=0)
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # Training loop variables
    time_step = 0
    i_episode = 0
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    # Main training loop
    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            current_agent = env.current_agent_idx
            action = ppo_agents[current_agent].select_action(state)
            state, reward, done, _ = env.step(action)

            ppo_agents[current_agent].buffer.rewards.append(reward)
            ppo_agents[current_agent].buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # Update if its time
            if time_step % update_timestep == 0:
                for agent in ppo_agents:
                    agent.update()

            # Decay action std if needed
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                for agent in ppo_agents:
                    agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Log if its time
            if time_step % log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            # Print if its time
            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                    i_episode, time_step, print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

            # Save model if its time
            if time_step % save_model_freq == 0:
                print("saving model at : " + checkpoint_path)
                for agent in ppo_agents:
                    agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

    log_f.close()
    env.close()

    # Print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
