import os
from datetime import datetime
import torch
import numpy as np
from models.ppo import PPO
from config.config import Config
from torch.utils.tensorboard import SummaryWriter
import json
from dataclasses import asdict
import platform
from utils.env_factory import make_env

def save_config_to_json(cfg: Config, writer_dir: str):
    """
    Save config to a JSON file in the same directory as tensorboard logs
    """
    # Convert Config dataclass to dict
    config_dict = asdict(cfg)
    
    # Add additional metadata
    metadata = {
        "config": config_dict,
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
    }
    
    # Create json file path
    json_path = os.path.join(writer_dir, 'config.json')
    
    # Save to JSON with nice formatting
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
# Add running reward stats
class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.count = 0
        self.eps = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        m_a = self.count * (self.std ** 2)
        m_b = batch_count * batch_var
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / (self.count + batch_count)
        self.std = np.sqrt(M2 / (self.count + batch_count))
        self.count += batch_count

    def normalize(self, x):
        return (x - self.mean) / (self.std + self.eps)

def train(cfg: Config, return_reward: bool = False):
    print("============================================================================================")

    # Create env using factory
    env, state_dim, action_dim = make_env(cfg)
    
    # Set up model saving
    if not os.path.exists(cfg.log.model_dir): 
        os.makedirs(cfg.log.model_dir)
    model_dir = os.path.join(cfg.log.model_dir, cfg.env.env_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
        
    run_num = len(next(os.walk(cfg.log.tensorboard_dir))[2])

    checkpoint_path = os.path.join(model_dir, 
                                 f"PPO_{cfg.env.env_name}_{cfg.ppo.random_seed}_{run_num}_{cfg.log.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

    # Create writer directory path
    writer_dir = os.path.join(cfg.log.tensorboard_dir, 
                           f"PPO_{cfg.env.env_name}_{cfg.ppo.random_seed}_{run_num}_{cfg.log.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    #set loggin dir to writer_dir
    log_dir = writer_dir
    log_f_name = os.path.join(log_dir, f"PPO_{cfg.env.env_name}_{cfg.ppo.random_seed}_{run_num}_{cfg.log.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Create writer
    writer = SummaryWriter(writer_dir)
    
    # Save config to JSON
    save_config_to_json(cfg, writer_dir)

    # Initialize agents with the writer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppo_agents = [
        PPO(state_dim=state_dim,
            action_dim=action_dim,
            cfg=cfg,
            writer=writer)
        for _ in range(env.num_agents)
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

    reward_normalizer = RunningMeanStd()
    
    # Start training loop
    start_time = datetime.now().replace(microsecond=0)
    while time_step <= cfg.env.max_training_timesteps:
        state_tuple = env.reset()
        observations = state_tuple[0]
        current_ep_reward = 0
        current_ep_length = 0

        while current_ep_length < cfg.env.max_ep_len:
            all_agents = list(env.agent_name_to_index.keys())
            actions = {}
            
            # Process each agent
            for agent_idx in all_agents:
                agent_index = env.agent_name_to_index[agent_idx]
                agent_state = observations[agent_idx]
                
                agent_state_tensor = torch.FloatTensor(agent_state).to(device)
                action = ppo_agents[agent_index].select_action(agent_state_tensor)
                
                if cfg.env.has_continuous_action_space:
                    action = action.flatten()
                else:
                    action = int(action)
                    
                actions[agent_idx] = action

            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Calculate average reward across all agents for this step
            step_reward = sum(rewards.values()) / len(rewards)
            current_ep_reward += step_reward  # Add average reward
            
            # Normalize rewards
            rewards_array = np.array(list(rewards.values()))
            reward_normalizer.update(rewards_array)
            normalized_rewards = reward_normalizer.normalize(rewards_array)
            
            # Update buffers with normalized rewards
            for agent_idx, reward in zip(rewards.keys(), normalized_rewards):
                agent_index = env.agent_name_to_index[agent_idx]
                agent = ppo_agents[agent_index]
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(terminations[agent_idx] or truncations[agent_idx])

            observations = next_observations
            time_step += 1
            current_ep_length += 1

            if all(terminations.values()) or all(truncations.values()):
                break

        # Calculate average episode reward
        current_ep_reward = current_ep_reward / current_ep_length  # Average over episode length
        
        # Update episode rewards list and print progress
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        
        # Update if its time
        if time_step % (cfg.env.max_ep_len * cfg.ppo.update_timestep) == 0:
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

        # After episode ends, add these lines:
        episode_avg_reward = current_ep_reward / time_step
        writer.add_scalar('Training/episode_reward', current_ep_reward, i_episode)
        writer.add_scalar('Training/episode_length', time_step, i_episode)
        writer.add_scalar('Training/average_reward', episode_avg_reward, i_episode)
        
        if cfg.env.has_continuous_action_space:
            writer.add_scalar('Policy/action_std', ppo_agents[0].action_std, i_episode)

        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

    final_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
    
    log_f.close()
    env.close()
    writer.close()
    
    if return_reward:
        return final_avg_reward

if __name__ == '__main__':
    cfg = Config()
    train(cfg)
    
    
    
    
    
    
    
