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

    # Start training loop
    start_time = datetime.now().replace(microsecond=0)
    while time_step <= cfg.env.max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        # Get list of all agents from the environment
        all_agents = list(env.agent_name_to_index.keys())  # Get all agent IDs
        
        # Initialize actions dictionary
        actions = {}
        
        # Process each agent
        for agent_idx in all_agents:
            agent_state = state.get(agent_idx, {})
            agent_index = env.agent_name_to_index[agent_idx]
            
            # Get the agent's state array from the first agent's state dictionary
            # since it contains all agents' states
            first_agent_state = state[all_agents[0]]  # Use the first agent's state dict
            if isinstance(first_agent_state.get(agent_idx), np.ndarray):
                agent_state_array = first_agent_state[agent_idx]
                agent_state_tensor = torch.FloatTensor(agent_state_array).to(device)
                action = ppo_agents[agent_index].select_action(agent_state_tensor)
            else:
                print(f"Warning: Empty state for {agent_idx}, using zero action")
                action = np.zeros(action_dim, dtype=np.float32)
                
            actions[agent_idx] = action

        # Step environment with all agents' actions
        try:
            step_result = env.step(actions)
            
            # Handle different return formats
            if len(step_result) == 4:
                next_state, rewards, dones, _ = step_result
            elif len(step_result) == 5:
                next_state, rewards, dones, _, _ = step_result
            else:
                print(f"Warning: Unexpected step result format: {len(step_result)} values")
                next_state, rewards, dones = step_result[:3]
                
            # Update buffers for each agent
            for agent_idx, reward in rewards.items():
                agent_index = env.agent_name_to_index[agent_idx]
                agent = ppo_agents[agent_index]
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(dones[agent_idx])

            state = next_state
            time_step += 1
            current_ep_reward += sum(rewards.values())

            # Check if episode is done (changed from dones[0] to check first agent)
            if dones[all_agents[0]]:  # Use the first agent's name instead of index
                break

        except Exception as e:
            print(f"Error during environment step: {e}")
            print(f"Actions provided: {actions}")
            raise

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

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

        # After episode ends, add these lines:
        episode_avg_reward = current_ep_reward / time_step
        writer.add_scalar('Training/episode_reward', current_ep_reward, i_episode)
        writer.add_scalar('Training/episode_length', time_step, i_episode)
        writer.add_scalar('Training/average_reward', episode_avg_reward, i_episode)
        
        if cfg.env.has_continuous_action_space:
            writer.add_scalar('Policy/action_std', ppo_agents[0].action_std, i_episode)

    final_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
    
    log_f.close()
    env.close()
    writer.close()
    
    if return_reward:
        return final_avg_reward

if __name__ == '__main__':
    cfg = Config()
    train(cfg)
    
    
    
    
    
    
    
