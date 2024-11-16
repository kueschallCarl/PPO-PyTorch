import os
from datetime import datetime
import torch
import numpy as np
from models.ppo import PPO
from utils.wrappers import PettingZooWrapper
from pettingzoo.mpe import simple_v3
from config.config import Config
from torch.utils.tensorboard import SummaryWriter

def train(cfg: Config):
    print("============================================================================================")

    # Create env
    raw_env = simple_v3.parallel_env(continuous_actions=cfg.env.continuous_actions)
    first_agent = raw_env.possible_agents[0]
    
    # Get state and action dimensions
    state_dim = raw_env.observation_space(first_agent).shape[0]
    if cfg.env.has_continuous_action_space:
        action_dim = raw_env.action_space(first_agent).shape[0]
    else:
        action_dim = raw_env.action_space(first_agent).n

    env = PettingZooWrapper(raw_env, num_agents=len(raw_env.possible_agents))

    # Set up logging
    if not os.path.exists(cfg.log.log_dir): 
        os.makedirs(cfg.log.log_dir)
    log_dir = os.path.join(cfg.log.log_dir, cfg.env.env_name)
    if not os.path.exists(log_dir): 
        os.makedirs(log_dir)
    
    run_num = len(next(os.walk(log_dir))[2])
    log_f_name = os.path.join(log_dir, f'PPO_{cfg.env.env_name}_log_{run_num}.csv')

    # Set up model saving
    if not os.path.exists(cfg.log.model_dir): 
        os.makedirs(cfg.log.model_dir)
    model_dir = os.path.join(cfg.log.model_dir, cfg.env.env_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    checkpoint_path = os.path.join(model_dir, 
                                 f"PPO_{cfg.env.env_name}_{cfg.ppo.random_seed}_{run_num}.pth")

    # Initialize agents
    ppo_agents = [
        PPO(state_dim=state_dim,
            action_dim=action_dim,
            cfg=cfg)
        for _ in range(len(raw_env.possible_agents))
    ]

    # Set random seed
    if cfg.ppo.random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", cfg.ppo.random_seed)
        torch.manual_seed(cfg.ppo.random_seed)
        env.seed(cfg.ppo.random_seed)
        np.random.seed(cfg.ppo.random_seed)

    # Logging
    print("Started training at (GMT) : ", datetime.now().replace(microsecond=0))
    print("============================================================================================")
    
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # Training loop variables
    time_step = 0
    i_episode = 0
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    # Add tensorboard writer for episode-level metrics
    writer = SummaryWriter(os.path.join(cfg.log.tensorboard_dir, 
                                       f"{cfg.env.env_name}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    # Start training loop
    start_time = datetime.now().replace(microsecond=0)
    while time_step <= cfg.env.max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, cfg.env.max_ep_len + 1):
            current_agent = env.current_agent_idx
            action = ppo_agents[current_agent].select_action(state)
            state, reward, done, _ = env.step(action)

            ppo_agents[current_agent].buffer.rewards.append(reward)
            ppo_agents[current_agent].buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # Update if its time
            if time_step % (cfg.env.max_ep_len * 4) == 0:
                for agent in ppo_agents:
                    agent.update()

            # Decay action std if needed
            if cfg.env.has_continuous_action_space and time_step % cfg.action.action_std_decay_freq == 0:
                for agent in ppo_agents:
                    agent.decay_action_std(cfg.action.action_std_decay_rate, 
                                         cfg.action.min_action_std)

            # Log if its time
            if time_step % cfg.log.log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            # Print if its time
            if time_step % cfg.log.print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                    i_episode, time_step, print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

            # Save model if its time
            if time_step % cfg.log.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                for agent in ppo_agents:
                    agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

        # After episode ends, add these lines:
        episode_avg_reward = current_ep_reward / t
        writer.add_scalar('Training/episode_reward', current_ep_reward, i_episode)
        writer.add_scalar('Training/episode_length', t, i_episode)
        writer.add_scalar('Training/average_reward', episode_avg_reward, i_episode)
        
        if cfg.env.has_continuous_action_space:
            writer.add_scalar('Policy/action_std', ppo_agents[0].action_std, i_episode)

    log_f.close()
    env.close()

    # Print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    # Close writers at the end
    writer.close()
    for agent in ppo_agents:
        agent.writer.close()

if __name__ == '__main__':
    cfg = Config()
    train(cfg)
    
    
    
    
    
    
    
