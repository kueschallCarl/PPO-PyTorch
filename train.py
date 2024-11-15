import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import pybullet_envs
from pettingzoo import ParallelEnv
from PPO import PPO
from pettingzoo.mpe import simple_v3


################################### Training ###################################

class PettingZooWrapper:
    def __init__(self, env_name, num_agents):
        self.env = env_name
        self.num_agents = num_agents
        self.agents = self.env.possible_agents
        self.reset()
        
    def reset(self):
        obs = self.env.reset()
        self.current_agent_idx = 0
        # Handle tuple return from parallel env
        if isinstance(obs, tuple):
            obs = obs[0]  # Get first element of tuple
        return obs[self.agents[0]]  # Return first agent's observation
        
    def step(self, action):
        agent = self.agents[self.current_agent_idx]
        # Create actions dict for all agents (others do nothing)
        actions = {a: action if a == agent else None for a in self.agents}
        
        # Parallel env step returns (obs, rewards, dones, truncated, infos)
        step_result = self.env.step(actions)
        obs, rewards, dones, truncated, infos = step_result
        
        # Debug print
        print(f"\nStep Debug:")
        print(f"Current agent: {agent}")
        print(f"Observations type: {type(obs)}")
        print(f"Observations keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        
        # Check if we need to reset
        if len(obs) == 0 or len(rewards) == 0 or len(dones) == 0:
            print("Environment needs reset - empty observation received")
            return self.reset(), 0.0, True, {}
            
        # Get current agent's values
        current_obs = obs[agent]
        current_reward = rewards[agent]
        current_done = dones[agent]
        current_info = infos[agent] if agent in infos else {}
        
        # Update agent index
        self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents
        
        return current_obs, current_reward, current_done, current_info
        
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()

def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "simple_v3"  # String name of the environment
    raw_env = simple_v3.parallel_env()  # Create the actual environment

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    # Create env
    raw_env = simple_v3.parallel_env()
    print("\nDebugging environment structure:")
    print(f"Possible agents: {raw_env.possible_agents}")
    print(f"Type of possible_agents: {type(raw_env.possible_agents)}")

    # Get first agent (with safety check)
    if len(raw_env.possible_agents) == 0:
        raise ValueError("No agents found in environment!")

    first_agent = raw_env.possible_agents[0]
    print(f"\nFirst agent: {first_agent}")

    # Debug observation space
    print("\nObservation space info:")
    obs_space = raw_env.observation_space(first_agent)
    print(f"Observation space: {obs_space}")
    print(f"Observation space type: {type(obs_space)}")

    # Debug action space
    print("\nAction space info:")
    act_space = raw_env.action_space(first_agent)
    print(f"Action space: {act_space}")
    print(f"Action space type: {type(act_space)}")

    # Set has_continuous_action_space based on the action space type
    has_continuous_action_space = isinstance(act_space, gym.spaces.Box)
    print(f"\nContinuous action space: {has_continuous_action_space}")

    env = PettingZooWrapper(raw_env, num_agents=len(raw_env.possible_agents))

    # state space dimension - get from first agent
    state_dim = raw_env.observation_space(first_agent).shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = raw_env.action_space(first_agent).shape[0]
    else:
        action_dim = raw_env.action_space(first_agent).n

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # Initialize separate PPO agents for each agent
    ppo_agents = [PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                      has_continuous_action_space, action_std) 
                  for _ in range(len(raw_env.possible_agents))]

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # Get current agent index
            current_agent = env.current_agent_idx
            
            # Select action for current agent
            action = ppo_agents[current_agent].select_action(state)
            state, reward, done, _ = env.step(action)

            # Save for current agent
            ppo_agents[current_agent].buffer.rewards.append(reward)
            ppo_agents[current_agent].buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # Update all agents
            if time_step % update_timestep == 0:
                for agent in ppo_agents:
                    agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                for agent in ppo_agents:
                    agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                for agent in ppo_agents:
                    agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
