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
from trainers.mappo_trainer import MAPPOTrainer

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

def train(
    cfg: Config, 
    return_reward: bool = False, 
    render: bool = False,
    pretrained_path: str = None,  # Path for fine-tuning
    checkpoint_path: str = None,  # Path for resuming training
):
    """
    Train PPO agents, with options for fresh training, resuming training, or fine-tuning.
    
    Args:
        cfg: Configuration object
        return_reward: Whether to return final average reward
        render: Whether to render environment during training
        pretrained_path: Path to pretrained model for fine-tuning (will modify learning rates)
        checkpoint_path: Path to checkpoint for resuming training (keeps original settings)
    """
    print("============================================================================================")

    # Create env using factory with optional rendering
    env, state_dim, action_dim = make_env(cfg, render_mode='human' if render else None)
    
    # Set up model saving
    if not os.path.exists(cfg.log.model_dir): 
        os.makedirs(cfg.log.model_dir)
    model_dir = os.path.join(cfg.log.model_dir, cfg.env.env_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
        
    run_num = len(next(os.walk(cfg.log.tensorboard_dir))[2])

    

    # Create writer directory path
    writer_dir = os.path.join(cfg.log.tensorboard_dir, 
                           f"PPO_{cfg.env.env_name}_{cfg.ppo.random_seed}_{run_num}_{cfg.log.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    trainer = MAPPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=env.num_agents,
        cfg=cfg,
        writer=writer
    )

    # Handle model loading for different scenarios
    if pretrained_path:
        print(f"Fine-tuning from pretrained models in: {pretrained_path}")
        # Load pretrained models
        load_checkpoint(trainer, pretrained_path)
            
        # Modify learning rates for fine-tuning
        for agent in trainer.agents:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.1  # Reduce learning rate for fine-tuning
                
        print("Loaded pretrained models and adjusted learning rates for fine-tuning")
        
    elif checkpoint_path:
        print(f"Resuming training from checkpoint directory: {checkpoint_path}")
        load_checkpoint(trainer, checkpoint_path)
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
        current_ep_reward = 0
        current_ep_length = 0

        # Reset environment
        observations = env.reset()[0]  # Get initial observations

        while True:
            # Select action with policy
            actions_dict = trainer.select_actions(observations)  # Returns dict of actions
            
            # Step environment with actions dictionary
            next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)
            
            # Store transition in buffer
            trainer.buffer.add(
                states=observations,
                actions=actions_dict,
                rewards=rewards,
                next_states=next_observations,
                dones={k: terminations[k] or truncations[k] for k in terminations.keys()}
            )

            # Update observations
            observations = next_observations
            
            # Calculate episode reward
            step_reward = sum(rewards.values()) / len(rewards)  # Average reward across agents
            current_ep_reward += step_reward
            
            time_step += 1
            current_ep_length += 1

            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                break

        # Calculate average episode reward
        current_ep_reward = current_ep_reward / current_ep_length  # Average over episode length
        
        # Update episode rewards list and print progress
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        
        # Log episode stats at the end of each episode
        trainer.log_episode_stats(current_ep_reward, current_ep_length)
        
        # Update if enough steps have been taken
        if time_step % (cfg.ppo.update_timestep * cfg.env.max_ep_len) == 0:
            trainer.update()

        # Decay action std if needed
        if cfg.env.has_continuous_action_space and time_step % cfg.action.action_std_decay_freq == 0:
            for agent in trainer.agents:
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
            save_checkpoint(trainer, model_dir_checkpoint, writer_dir_checkpoint)
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
            writer.add_scalar('Policy/action_std', trainer.agents[0].action_std, i_episode)

        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

    final_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
    
    log_f.close()
    env.close()
    writer.close()
    
    # Save final model
    print("Saving final model...")
    save_checkpoint(trainer, model_dir_checkpoint, writer_dir_checkpoint)
    print("Final model saved at:")
    print(f"- {model_dir_checkpoint}")
    print(f"- {writer_dir_checkpoint}")

    if return_reward:
        return final_avg_reward

def save_checkpoint(trainer, model_dir_checkpoint, writer_dir_checkpoint):
    """
    Save agent checkpoints in both locations:
    - model_dir: Full path with timestamp etc.
    - writer_dir: Simple 'model_agentX.pth' in the run directory
    """
    # Save in model_dir (archive)
    for agent_idx, agent in enumerate(trainer.agents):
        model_path = model_dir_checkpoint.replace('.pth', f'_agent{agent_idx}.pth')
        agent.save(model_path)
    
    # Save in writer_dir (run directory)
    for agent_idx, agent in enumerate(trainer.agents):
        writer_path = os.path.join(os.path.dirname(writer_dir_checkpoint), f'model_agent{agent_idx}.pth')
        agent.save(writer_path)

def load_checkpoint(trainer, checkpoint_dir):
    """
    Load agent checkpoints from a run directory
    checkpoint_dir: path to the run directory containing model_agent{X}.pth files
    """
    for agent_idx, agent in enumerate(trainer.agents):
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
    
    
    
    
    
    
    
