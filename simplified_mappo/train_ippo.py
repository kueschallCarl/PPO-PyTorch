import os
from datetime import datetime
import torch
import numpy as np
from algorithms.ppo import PPO
from config.config import Config
from torch.utils.tensorboard import SummaryWriter
import json
from dataclasses import asdict
import platform
from envs.mpe.scenarios import SCENARIOS
import envs.mpe.scenarios as scenarios
import envs.mpe.core as core

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

def make_env(cfg, render_mode=None):
    """
    Create a Multi-Agent Particle Environment (MPE) using local implementation
    """
    # Get the scenario class from the scenarios
    scenario = SCENARIOS[cfg.env.env_name.split('_v')[0]]()
    
    # Create world
    world = scenario.make_world()
    
    # Get dimensions
    obs_dim = len(scenario.observation(world.agents[0], world))
    if cfg.env.has_continuous_action_space:
        action_dim = world.dim_p  # Physical action dimension
    else:
        # For discrete actions, would need to be adjusted based on your needs
        action_dim = 5  # Example: 5 discrete actions
    
    return world, obs_dim, action_dim, scenario

def train(
    cfg: Config, 
    return_reward: bool = False, 
    render: bool = False,
    pretrained_path: str = None,
    checkpoint_path: str = None,
):
    """
    Train PPO agents using local MPE implementation
    """
    print("============================================================================================")

    # Create env using local implementation
    world, state_dim, action_dim, scenario = make_env(cfg, render_mode='human' if render else None)
    
    # Set up model saving - create directories if they don't exist
    if not os.path.exists(cfg.log.model_dir): 
        os.makedirs(cfg.log.model_dir)
    model_dir = os.path.join(cfg.log.model_dir, cfg.env.env_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
        
    # Create tensorboard directory if it doesn't exist
    if not os.path.exists(cfg.log.tensorboard_dir):
        os.makedirs(cfg.log.tensorboard_dir)
        
    # Safely get run number
    try:
        run_num = len([d for d in os.listdir(cfg.log.tensorboard_dir) 
                      if os.path.isfile(os.path.join(cfg.log.tensorboard_dir, d))])
    except (FileNotFoundError, StopIteration):
        run_num = 0

    # Create writer directory path
    writer_dir = os.path.join(cfg.log.tensorboard_dir, 
                           f"PPO_{cfg.env.env_name}_{cfg.ppo.random_seed}_{run_num}_{cfg.log.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create writer directory if it doesn't exist
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
        
    # Create new checkpoint paths for both locations
    checkpoint_filename = f"PPO_{cfg.env.env_name}_{cfg.log.run_name}_{cfg.ppo.random_seed}_{run_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    model_dir_checkpoint = os.path.join(model_dir, checkpoint_filename)
    writer_dir_checkpoint = os.path.join(writer_dir, "model.pth")
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
        for _ in range(len(world.agents))
    ]

    # Handle model loading for different scenarios
    if pretrained_path:
        print(f"Fine-tuning from pretrained models in: {pretrained_path}")
        # Load pretrained models
        load_checkpoint(ppo_agents, pretrained_path)
            
        # Modify learning rates for fine-tuning
        for agent in ppo_agents:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.1  # Reduce learning rate for fine-tuning
                
        print("Loaded pretrained models and adjusted learning rates for fine-tuning")
        
    elif checkpoint_path:
        print(f"Resuming training from checkpoint directory: {checkpoint_path}")
        load_checkpoint(ppo_agents, checkpoint_path)
        print("Resumed from checkpoint successfully")

    # Set initial random seed if specified
    if cfg.ppo.random_seed is not None:
        print("--------------------------------------------------------------------------------------------")
        print("setting initial random seed to ", cfg.ppo.random_seed)
        torch.manual_seed(cfg.ppo.random_seed)
        np.random.seed(cfg.ppo.random_seed)
    
    # Logging
    print("Started training at (GMT) : ", datetime.now().replace(microsecond=0))
    print("============================================================================================")
    
    # Create log file path
    log_f_name = os.path.join(writer_dir, 'training_log.csv')
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
        episode_seed = np.random.randint(0, 10000)
        
        # Reset world using scenario instance
        scenario.reset_world(world)
        observations = {f'agent_{i}': scenario.observation(agent, world) 
                       for i, agent in enumerate(world.agents)}
        
        current_ep_reward = 0
        current_ep_length = 0
        
        while current_ep_length < cfg.env.max_ep_len:
            actions = {}
            
            # Get actions for each agent
            for i, agent in enumerate(world.agents):
                agent_obs = observations[f'agent_{i}']
                agent_state_tensor = torch.FloatTensor(agent_obs).to(device)
                action = ppo_agents[i].select_action(agent_state_tensor)
                
                if cfg.env.has_continuous_action_space:
                    action = action.flatten()
                    agent.action.u = action  # Set physical action
                else:
                    action = int(action)
                    # Would need to convert discrete action to continuous for MPE
                
                actions[f'agent_{i}'] = action
            
            # Step world
            world.step()
            
            # Get new observations and rewards using scenario instance
            next_observations = {f'agent_{i}': scenario.observation(agent, world) 
                               for i, agent in enumerate(world.agents)}
            rewards = {f'agent_{i}': scenario.reward(agent, world) 
                      for i, agent in enumerate(world.agents)}
            
            # Calculate if episode should terminate
            # You might want to add your own termination conditions
            terminations = {f'agent_{i}': False for i in range(len(world.agents))}
            truncations = {f'agent_{i}': current_ep_length >= cfg.env.max_ep_len - 1 
                          for i in range(len(world.agents))}
            
            # Calculate average reward across all agents for this step
            step_reward = sum(rewards.values()) / len(rewards)
            current_ep_reward += step_reward  # Add average reward
            
            # Normalize rewards
            rewards_array = np.array(list(rewards.values()))
            reward_normalizer.update(rewards_array)
            normalized_rewards = reward_normalizer.normalize(rewards_array)
            
            # Update buffers with normalized rewards
            for i, (agent_idx, reward) in enumerate(zip(rewards.keys(), normalized_rewards)):
                agent = ppo_agents[i]  # Use index directly since it matches the agent order
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
            print("saving model checkpoints...")
            save_checkpoint(ppo_agents, model_dir_checkpoint, writer_dir_checkpoint)
            print("models saved at:")
            print(f"- {model_dir_checkpoint}")
            print(f"- {writer_dir_checkpoint}")
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
    
    # Save final model
    print("Saving final model...")
    save_checkpoint(ppo_agents, model_dir_checkpoint, writer_dir_checkpoint)
    print("Final model saved at:")
    print(f"- {model_dir_checkpoint}")
    print(f"- {writer_dir_checkpoint}")

    if return_reward:
        return final_avg_reward

def save_checkpoint(agents, model_dir_checkpoint, writer_dir_checkpoint):
    """
    Save agent checkpoints in both locations:
    - model_dir: Full path with timestamp etc.
    - writer_dir: Simple 'model_agentX.pth' in the run directory
    """
    # Save in model_dir (archive)
    for agent_idx, agent in enumerate(agents):
        model_path = model_dir_checkpoint.replace('.pth', f'_agent{agent_idx}.pth')
        agent.save(model_path)
    
    # Save in writer_dir (run directory)
    for agent_idx, agent in enumerate(agents):
        writer_path = os.path.join(os.path.dirname(writer_dir_checkpoint), f'model_agent{agent_idx}.pth')
        agent.save(writer_path)

def load_checkpoint(agents, checkpoint_dir):
    """
    Load agent checkpoints from a run directory
    checkpoint_dir: path to the run directory containing model_agent{X}.pth files
    """
    for agent_idx, agent in enumerate(agents):
        agent_checkpoint = os.path.join(checkpoint_dir, f'model_agent{agent_idx}.pth')
        if not os.path.exists(agent_checkpoint):
            raise FileNotFoundError(f"Model for agent {agent_idx} not found at: {agent_checkpoint}")
        agent.load(agent_checkpoint)

if __name__ == '__main__':
    cfg = Config()
    
    # Example of fine-tuning a pretrained model
    pretrained_model = "runs/PPO_simple_v3_None_0_fixing_IPPO_20241123_224202"  # Directory path, not file path    
    # Verify file exists before starting
    if pretrained_model:
        if not os.path.exists(pretrained_model):
            print(f"Error: Pretrained model not found at {pretrained_model}")
            # List available models
            model_dir = "logs/PPO_preTrained/simple_v3/"
            if os.path.exists(model_dir):
                print("\nAvailable models:")
                for file in os.listdir(model_dir):
                    if file.endswith(".pth"):
                        print(f"- {file}")
    # Start fine-tuning with rendering enabled
    train(
        cfg, 
        pretrained_path=None,
        render=False
        )
    
    
    
    
    
    
    