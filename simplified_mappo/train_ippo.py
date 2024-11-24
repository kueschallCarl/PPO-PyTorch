import os
from datetime import datetime
import torch
import numpy as np
from algorithms.ppo import PPO
from config.config import Config
import json
from dataclasses import asdict
import platform
from envs.mpe.scenarios import SCENARIOS
import envs.mpe.scenarios as scenarios
import envs.mpe.core as core
import argparse
import wandb
import logging
import traceback
from utils.visualization import render_env
import matplotlib.pyplot as plt

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
    scenario = SCENARIOS[cfg.env.env_name]()
    
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

def save_checkpoint(agents, model_dir_checkpoint, writer_dir):
    """
    Save agent checkpoints in both locations:
    - model_dir: Archive location with timestamp
    - writer_dir: Run directory for easy loading
    """
    # Save in model_dir (archive)
    for agent_idx, agent in enumerate(agents):
        model_path = model_dir_checkpoint.replace('.pth', f'_agent{agent_idx}.pth')
        agent.save(model_path)
    
    # Save in writer_dir (run directory)
    for agent_idx, agent in enumerate(agents):
        writer_path = os.path.join(writer_dir, f'model_agent{agent_idx}.pth')
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

def train_ippo(
    cfg: Config, 
    return_reward: bool = False, 
    render: bool = False,
    pretrained_path: str = None,
    checkpoint_path: str = None,
):
    """
    Train PPO agents using local MPE implementation
    """
    # Initialize global_step at the start of the function
    global_step = 0
    
    # Initialize final_avg_reward at the start
    final_avg_reward = 0
    
    # Initialize wandb if enabled
    if cfg.log.use_wandb:
        wandb.init(
            project=cfg.log.wandb_project,
            entity=cfg.log.wandb_entity,
            name=cfg.log.run_name,
            config=asdict(cfg)
        )

    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(cfg.log.log_dir, f'training_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )

    try:
        print("============================================================================================")
        logging.info(f"Starting training with config:")
        logging.info(f"Environment: {cfg.env.env_name}")
        logging.info(f"Max episodes: {cfg.env.max_episodes}")

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
                               f"PPO_{cfg.env.env_name}_{cfg.seed}_{run_num}_{cfg.log.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create writer directory if it doesn't exist
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
            
        # Create new checkpoint paths for both locations
        checkpoint_filename = f"PPO_{cfg.env.env_name}_{cfg.log.run_name}_{cfg.seed}_{run_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        model_dir_checkpoint = os.path.join(model_dir, checkpoint_filename)
        writer_dir_checkpoint = os.path.join(writer_dir, "model.pth")
        # Create writer
        
        # Save config to JSON
        save_config_to_json(cfg, writer_dir)

        # Initialize agents without writer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agents = [
            PPO(state_dim=state_dim,
                action_dim=action_dim,
                cfg=cfg)
            for _ in range(len(world.agents))
        ]

        # Handle model loading for different scenarios
        if pretrained_path:
            print(f"Fine-tuning from pretrained models in: {pretrained_path}")
            logging.info("Loaded pretrained models and adjusted learning rates for fine-tuning")
            
        elif checkpoint_path:
            print(f"Resuming training from checkpoint directory: {checkpoint_path}")
            load_checkpoint(ppo_agents, checkpoint_path)
            print("Resumed from checkpoint successfully")

        # Set initial random seed if specified
        if cfg.seed is not None:
            print("--------------------------------------------------------------------------------------------")
            print("setting initial random seed to ", cfg.seed)
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
        
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
        
        # Initialize global step counter
        global_step = 0
        
        # Start training loop
        start_time = datetime.now().replace(microsecond=0)
        for i_episode in range(cfg.env.max_episodes):
            episode_seed = np.random.randint(0, 10000)
            
            # Reset world using scenario instance
            scenario.reset_world(world)
            observations = {f'agent_{i}': scenario.observation(agent, world) 
                           for i, agent in enumerate(world.agents)}
            
            current_ep_reward = 0
            current_ep_length = 0
            
            while current_ep_length < cfg.env.episode_length:
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
                truncations = {f'agent_{i}': current_ep_length >= cfg.env.episode_length - 1 
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
                global_step += 1  # Keep tracking global steps for logging

                # Log step metrics to wandb with global step
                if cfg.log.use_wandb and time_step % cfg.log.log_freq == 0:
                    step_metrics = {
                        "training/step_reward": step_reward,
                        "training/timestep": time_step,
                        "training/episode": i_episode,
                    }
                    wandb.log(step_metrics, step=global_step)

                if all(terminations.values()) or all(truncations.values()):
                    break

            # Calculate average episode reward
            current_ep_reward = current_ep_reward / current_ep_length  # Average over episode length
            
            # Update episode rewards list and print progress
            print_running_reward += current_ep_reward
            print_running_episodes += 1
            
            # Update if its time
            if time_step % (cfg.env.episode_length * cfg.training.eval_frequency) == 0:
                for agent in ppo_agents:
                    agent.update()

            # Decay action std if needed
            if cfg.env.has_continuous_action_space and time_step % cfg.training.action_std_decay_freq == 0:
                for agent in ppo_agents:
                    agent.decay_action_std(cfg.training.action_std_decay_rate, 
                                         cfg.training.min_action_std)

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
            if global_step % cfg.log.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model checkpoints...")
                save_checkpoint(ppo_agents, model_dir_checkpoint, writer_dir)
                print("models saved in:")
                print(f"- {model_dir}")
                print(f"- {writer_dir}")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                
                if cfg.log.use_wandb:
                    for agent_idx, agent in enumerate(ppo_agents):
                        model_path = os.path.join(writer_dir, f'model_agent{agent_idx}.pth')
                        wandb.save(model_path)
                        wandb.run.summary[f"agent{agent_idx}_model_step_{global_step}"] = model_path

            # After episode ends, add these lines:
            episode_avg_reward = current_ep_reward / time_step
            log_running_reward += current_ep_reward
            log_running_episodes += 1
            i_episode += 1

            # After episode ends, add wandb logging
            if cfg.log.use_wandb:
                # Per-step metrics
                step_metrics = {
                    # Environment state
                    "env/episode_progress": current_ep_length / cfg.env.episode_length,
                    "env/total_episodes": i_episode,
                    "env/steps_remaining": cfg.env.max_episodes - time_step,
                    
                    # Reward tracking
                    "rewards/step_reward_raw": step_reward,
                    "rewards/step_reward_normalized": reward_normalizer.normalize(np.array([step_reward]))[0],
                    "rewards/running_mean": reward_normalizer.mean,
                    "rewards/running_std": reward_normalizer.std,
                    
                    # Per-agent metrics
                    **{f"agent_{i}/raw_reward": reward for i, reward in enumerate(rewards.values())},
                    **{f"agent_{i}/normalized_reward": norm_reward for i, norm_reward in enumerate(normalized_rewards)},
                    **{f"agent_{i}/buffer_size": len(ppo_agents[i].buffer.rewards) for i in range(len(ppo_agents))},
                }
                
                # Add action statistics for each agent
                for i, agent in enumerate(ppo_agents):
                    if cfg.env.has_continuous_action_space:
                        actions_array = np.array([actions[f'agent_{i}']])
                        actions_tensor = torch.from_numpy(actions_array)
                        step_metrics.update({
                            f"agent_{i}/action_mean": actions_tensor.mean().item(),
                            f"agent_{i}/action_std": actions_tensor.std().item(),
                            f"agent_{i}/action_max": actions_tensor.max().item(),
                            f"agent_{i}/action_min": actions_tensor.min().item(),
                            f"agent_{i}/current_action_std": agent.action_std,
                        })
                
                wandb.log(step_metrics, step=global_step)

            # After episode ends, update episode metrics:
            if cfg.log.use_wandb:
                episode_metrics = {
                    # Episode statistics
                    "episode/total_reward": current_ep_reward,
                    "episode/length": current_ep_length,
                    "episode/average_reward": current_ep_reward / current_ep_length,
                    "episode/normalized_reward": reward_normalizer.normalize(np.array([current_ep_reward]))[0],
                    
                    # Training progress
                    "training/episodes_completed": i_episode,
                    "training/total_timesteps": time_step,
                    "training/completion_percentage": (time_step / cfg.env.max_episodes) * 100,
                    
                    # Running statistics
                    "training/running_reward": print_running_reward / max(print_running_episodes, 1),
                    "training/running_length": current_ep_length,
                    
                    # Learning rates
                    **{f"agent_{i}/lr_actor": agent.optimizer.param_groups[0]['lr'] for i, agent in enumerate(ppo_agents)},
                    **{f"agent_{i}/lr_critic": agent.optimizer.param_groups[1]['lr'] for i, agent in enumerate(ppo_agents)},
                }
                
                wandb.log(episode_metrics, step=global_step)

            # During model saving
            if time_step % cfg.log.save_model_freq == 0:
                if cfg.log.use_wandb:
                    # Log model checkpoints to wandb
                    for agent_idx, agent in enumerate(ppo_agents):
                        model_path = model_dir_checkpoint.replace('.pth', f'_agent{agent_idx}.pth')
                        wandb.save(model_path)
                        # Use global step in summary
                        wandb.run.summary[f"agent{agent_idx}_model_step_{global_step}"] = model_path

            # Evaluate policy periodically
            if (i_episode + 1) % cfg.training.eval_frequency == 0:
                eval_rewards = []
                # Run multiple evaluation episodes
                for eval_ep in range(5):  # Run 5 evaluation episodes
                    scenario.reset_world(world)
                    eval_ep_reward = 0
                    
                    # Create figure for visualization if needed
                    if cfg.training.visualize_eval and eval_ep == 0:  # Only visualize first episode
                        plt.figure(figsize=(8, 8))
                    
                    # Run one evaluation episode
                    for step in range(cfg.env.episode_length):
                        actions = {}
                        for i, agent in enumerate(world.agents):
                            agent_obs = scenario.observation(agent, world)
                            agent_state_tensor = torch.FloatTensor(agent_obs).to(device)
                            # Use deterministic action selection for evaluation
                            action = ppo_agents[i].select_action(agent_state_tensor, deterministic=True)
                            actions[f'agent_{i}'] = action
                            agent.action.u = action  # Set physical action
                        
                        # Store previous positions for interpolation if visualizing
                        if cfg.training.visualize_eval and eval_ep == 0:
                            prev_positions = np.array([agent.state.p_pos.copy() for agent in world.agents])
                            prev_velocities = np.array([agent.state.p_vel.copy() for agent in world.agents])
                        
                        # Step world
                        world.step()
                        
                        # Visualize if needed
                        if cfg.training.visualize_eval and eval_ep == 0:
                            render_env(world)
                            plt.pause(cfg.training.eval_delay)
                        
                        # Get rewards
                        rewards = {f'agent_{i}': scenario.reward(agent, world) 
                                 for i, agent in enumerate(world.agents)}
                        eval_ep_reward += sum(rewards.values()) / len(rewards)
                    
                    eval_rewards.append(eval_ep_reward)
                    
                    # Close visualization for this episode
                    if cfg.training.visualize_eval and eval_ep == 0:
                        plt.close()
                
                # Calculate average evaluation reward
                avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
                
                if cfg.log.use_wandb:
                    eval_metrics = {
                        "eval/reward": avg_eval_reward,
                        "eval/reward_diff": avg_eval_reward - current_ep_reward,
                        "eval/reward_std": np.std(eval_rewards)
                    }
                    wandb.log(eval_metrics, step=global_step)
                
                print(f"Evaluation at episode {i_episode + 1}: Average Reward = {avg_eval_reward:.2f}")

        # Calculate final average reward
        final_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
        
        log_f.close()
        
        # Save final models
        print("Saving final models...")
        save_checkpoint(ppo_agents, model_dir_checkpoint, writer_dir)
        print("Final models saved in:")
        print(f"- {model_dir}")
        print(f"- {writer_dir}")
        
        if cfg.log.use_wandb:
            for agent_idx, agent in enumerate(ppo_agents):
                model_path = os.path.join(writer_dir, f'model_agent{agent_idx}.pth')
                wandb.save(model_path)
                wandb.run.summary[f"agent{agent_idx}_final_model"] = model_path

        if return_reward:
            return final_avg_reward

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        logging.error(traceback.format_exc())
        raise

    finally:
        if cfg.log.use_wandb:
            # Use final global step in summary
            wandb.run.summary.update({
                "final_avg_reward": final_avg_reward,
                "total_timesteps": global_step,
                "total_episodes": i_episode if 'i_episode' in locals() else 0
            })
            wandb.finish()

        # Safely close files if they exist
        if 'log_f' in locals() and not log_f.closed:
            log_f.close()

    if return_reward:
        return final_avg_reward

    
    
    
    
    
    
    