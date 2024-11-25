import torch
from algorithms.actor_critic import MLPPolicy
from algorithms.mappo import MAPPO
from utils.buffer import SharedReplayBuffer
from runner import Runner
import logging
import time
from datetime import datetime
import traceback
import wandb
from dataclasses import asdict, dataclass, field
from typing import Optional, List
import os
from config.config import Config
import numpy as np
from utils.running_mean_std import RunningMeanStd
from utils.visualization import render_env
import matplotlib.pyplot as plt

def train_mappo(cfg: Config):
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
        logging.info(f"Starting training with config:")
        logging.info(f"Scenario: {cfg.env.env_name}, Agents: {cfg.env.num_agents}")
        
        # Initialize global step counter
        global_step = 0
        
        # Initialize device
        device = torch.device(cfg.device)
        logging.info(f"Using device: {device}")
        
        # Initialize environment and runner
        runner = Runner(cfg.env.env_name, cfg.env.num_agents, device=device)
        
        # Initialize policy and algorithm
        policy = MLPPolicy(
            obs_space=runner.obs_space,
            action_space=runner.action_space
        ).to(device)
        
        algorithm = MAPPO(
            policy=policy,
            cfg=cfg
        )
        
        # Initialize buffer
        buffer = SharedReplayBuffer(
            num_agents=cfg.env.num_agents,
            obs_space=runner.obs_space,
            act_space=runner.action_space,
            size=cfg.buffer.size,
            device=device
        )
        
        # Initialize reward normalizer
        reward_normalizer = RunningMeanStd()
        
        # Training loop
        start_time = datetime.now()
        print_running_reward = 0
        print_running_episodes = 0
        log_running_reward = 0
        log_running_episodes = 0
        
        # Create writer directory path
        writer_dir = os.path.join(cfg.log.tensorboard_dir, 
                               f"PPO_{cfg.env.env_name}_{cfg.seed}_{cfg.log.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create writer directory if it doesn't exist
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
        
        # Create checkpoint paths
        model_checkpoint = os.path.join(writer_dir, "model.pth")
        
        for episode in range(cfg.env.max_episodes):
            episode_start = time.time()
            
            try:
                # Collect experience
                episode_reward = runner.collect_episodes(policy, buffer, cfg.env.episode_length)
                global_step += cfg.env.episode_length
                
                # Update running rewards
                print_running_reward += episode_reward
                print_running_episodes += 1
                log_running_reward += episode_reward
                log_running_episodes += 1
                
                # Track training metrics
                if cfg.log.use_wandb:
                    # Environment state
                    elapsed_time = datetime.now() - start_time
                    step_metrics = {
                        "env/episode": episode,
                        "env/steps_total": global_step,
                        "env/episode_length": cfg.env.episode_length,
                        "env/episode_progress": global_step / cfg.env.max_training_timesteps,
                        
                        # Reward tracking
                        "rewards/episode_reward": episode_reward,
                        "rewards/episode_reward_mean": episode_reward / cfg.env.episode_length,
                        "rewards/running_mean": reward_normalizer.mean,
                        "rewards/running_std": reward_normalizer.std,
                        
                        # Time tracking (convert to seconds for logging)
                        "time/episode_duration": time.time() - episode_start,
                        "time/total_duration": elapsed_time.total_seconds(),
                        
                        # Training progress
                        "training/episodes_completed": episode,
                        "training/total_timesteps": global_step,
                        "training/completion_percentage": (global_step / cfg.env.max_training_timesteps) * 100,
                        "training/running_reward": print_running_reward / max(print_running_episodes, 1),
                        "training/running_length": cfg.env.episode_length,
                    }
                    
                    wandb.log(step_metrics, step=global_step)
                
                # Update policy
                policy_metrics = {}
                for sample in buffer.get_samples(cfg.buffer.batch_size):
                    update_info = algorithm.update(sample, global_step)
                    # Aggregate policy update metrics
                    for k, v in update_info.items():
                        if k not in policy_metrics:
                            policy_metrics[k] = []
                        policy_metrics[k].append(v)
                
                # Average and log policy metrics
                if cfg.log.use_wandb:
                    avg_policy_metrics = {
                        f"policy/{k}": np.mean(v) for k, v in policy_metrics.items()
                    }
                    wandb.log(avg_policy_metrics, step=global_step)
                
                # Print progress
                if (episode + 1) % cfg.log.print_freq == 0:
                    print_avg_reward = print_running_reward / print_running_episodes
                    print(f"Episode {episode + 1} \t Steps {global_step} \t Average Reward {print_avg_reward:.2f}")
                    print_running_reward = 0
                    print_running_episodes = 0
                
                # Evaluate policy
                if (episode + 1) % cfg.training.eval_frequency == 0:
                    try:
                        if cfg.training.visualize_eval:
                            try:
                                eval_reward, position_history = runner.eval_policy(
                                    policy, 
                                    visualize=True,
                                    eval_delay=cfg.training.eval_delay,
                                    eval_episode_length=cfg.env.episode_length,
                                )
                            except Exception as e:
                                logging.error(f"Visualization failed: {str(e)}")
                                # Fallback to non-visual evaluation
                                eval_reward = runner.eval_policy(
                                    policy,
                                    visualize=False,
                                    eval_episode_length=cfg.env.episode_length,
                                )
                                position_history = None
                            
                            # Log additional metrics if using wandb
                            if cfg.log.use_wandb:
                                # Calculate average distances over episode
                                avg_distances = np.mean([
                                    [min(agent_distances) for agent_distances in step_distances]
                                    for step_distances in position_history['distances']
                                ], axis=0)
                                
                                eval_metrics = {
                                    "eval/reward": eval_reward,
                                    **{f"eval/agent_{i}_avg_distance": dist for i, dist in enumerate(avg_distances)},
                                    "eval/max_distance": np.max([max(d) for d in position_history['distances']]),
                                    "eval/min_distance": np.min([min(d) for d in position_history['distances']])
                                }
                                wandb.log(eval_metrics, step=global_step)
                        else:
                            eval_reward = runner.eval_policy(
                                policy, 
                                visualize=False
                            )
                            if cfg.log.use_wandb:
                                eval_metrics = {
                                    "eval/reward": eval_reward,
                                }
                                wandb.log(eval_metrics, step=global_step)
                                
                    except Exception as e:
                        logging.error(f"Error during evaluation: {str(e)}")
                        logging.error(traceback.format_exc())
                        continue
            
            except Exception as e:
                logging.error(f"Error during episode {episode + 1}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        # Log final metrics
        if cfg.log.use_wandb:
            elapsed_time = datetime.now() - start_time
            wandb.run.summary.update({
                "training/total_episodes": cfg.env.max_episodes,
                "training/total_steps": global_step,
                "training/total_time": elapsed_time.total_seconds(),
                "training/final_eval_reward": eval_reward if 'eval_reward' in locals() else None,
                "training/final_running_reward": log_running_reward / max(log_running_episodes, 1)
            })
        
        # Save model periodically
        if global_step % cfg.log.save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model checkpoint...")
            torch.save(policy.state_dict(), model_checkpoint)
            print("model saved at:", model_checkpoint)
            print("Elapsed Time  : ", datetime.now() - start_time)
            print("--------------------------------------------------------------------------------------------")
            
            if cfg.log.use_wandb:
                wandb.save(model_checkpoint)
                wandb.run.summary[f"model_step_{global_step}"] = model_checkpoint
        
        # Save final model
        print("Saving final model...")
        torch.save(policy.state_dict(), model_checkpoint)
        print("Final model saved at:", model_checkpoint)
        
        if cfg.log.use_wandb:
            wandb.save(model_checkpoint)
            wandb.run.summary["final_model"] = model_checkpoint
        
    except Exception as e:
        logging.error(f"Fatal error during training: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    finally:
        if cfg.log.use_wandb:
            wandb.finish() 