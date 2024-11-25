import torch
from algorithms.ippo_actor_critic import IPPOPolicy
from algorithms.ippo import IPPO
from utils.buffer import SharedReplayBuffer
from runner_ippo import RunnerIPPO
import logging
import time
from datetime import datetime
import traceback
import wandb
from dataclasses import asdict
import os
from config.config import Config
import numpy as np

def train_ippo(cfg: Config):
    # Initialize wandb if enabled
    if cfg.log.use_wandb:
        wandb.init(
            project=cfg.log.wandb_project,
            entity=cfg.log.wandb_entity,
            name=cfg.log.run_name,
            config=asdict(cfg)
        )
    
    # Create log directory if it doesn't exist
    os.makedirs(cfg.log.log_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(cfg.log.log_dir, f'training_ippo_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )
    
    try:
        logging.info(f"Starting IPPO training with config:")
        logging.info(f"Scenario: {cfg.env.env_name}, Agents: {cfg.env.num_agents}")
        
        # Initialize global step counter
        global_step = 0
        
        # Initialize device
        device = torch.device(cfg.device)
        logging.info(f"Using device: {device}")
        
        # Initialize environment and runner
        runner = RunnerIPPO(cfg.env.env_name, cfg.env.num_agents, device=device)
        
        # Initialize policies (one per agent)
        policies = [
            IPPOPolicy(
                obs_space=runner.obs_space,
                action_space=runner.action_space
            ).to(device) for _ in range(cfg.env.num_agents)
        ]
        
        # Initialize algorithms (one per agent)
        algorithms = [
            IPPO(
                policy=policy,
                cfg=cfg
            ) for policy in policies
        ]
        
        # Initialize buffers (one per agent)
        buffers = [
            SharedReplayBuffer(
                num_agents=1,  # Each buffer is for one agent
                obs_space=runner.obs_space,
                act_space=runner.action_space,
                size=cfg.buffer.size,
                device=device
            ) for _ in range(cfg.env.num_agents)
        ]
        
        # Training loop
        start_time = datetime.now()
        print_running_reward = 0
        print_running_episodes = 0
        
        for episode in range(cfg.env.max_episodes):
            episode_start = time.time()
            
            try:
                # Collect experience
                episode_reward = runner.collect_episodes(
                    policies, buffers, cfg.env.episode_length
                )
                global_step += cfg.env.episode_length
                
                # Update running rewards
                print_running_reward += episode_reward
                print_running_episodes += 1
                
                # Update policies
                policy_metrics = {f"agent_{i}": {} for i in range(cfg.env.num_agents)}
                
                for agent_id in range(cfg.env.num_agents):
                    for sample in buffers[agent_id].get_samples(cfg.buffer.batch_size):
                        update_info = algorithms[agent_id].update(sample, global_step)
                        
                        # Aggregate policy update metrics
                        for k, v in update_info.items():
                            if k not in policy_metrics[f"agent_{agent_id}"]:
                                policy_metrics[f"agent_{agent_id}"][k] = []
                            policy_metrics[f"agent_{agent_id}"][k].append(v)
                
                # Log metrics
                if cfg.log.use_wandb:
                    metrics = {
                        "env/episode": episode,
                        "env/steps_total": global_step,
                        "env/episode_reward": episode_reward,
                        "time/episode_duration": time.time() - episode_start,
                    }
                    
                    # Add policy metrics
                    for agent_id in range(cfg.env.num_agents):
                        for k, v in policy_metrics[f"agent_{agent_id}"].items():
                            metrics[f"agent_{agent_id}/{k}"] = np.mean(v)
                    
                    wandb.log(metrics, step=global_step)
                
                # Print progress
                if (episode + 1) % cfg.log.print_freq == 0:
                    print_avg_reward = print_running_reward / print_running_episodes
                    print(f"Episode {episode + 1} \t Steps {global_step} \t Average Reward {print_avg_reward:.2f}")
                    print_running_reward = 0
                    print_running_episodes = 0
                
                # Evaluate policies
                if (episode + 1) % cfg.training.eval_frequency == 0:
                    eval_reward = runner.eval_policy(
                        policies,
                        visualize=cfg.training.visualize_eval,
                        eval_delay=cfg.training.eval_delay
                    )
                    
                    if cfg.log.use_wandb:
                        wandb.log({"eval/reward": eval_reward}, step=global_step)
            
            except Exception as e:
                logging.error(f"Error during episode {episode + 1}: {str(e)}")
                logging.error(traceback.format_exc())
                continue
        
    except Exception as e:
        logging.error(f"Fatal error during training: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    finally:
        if cfg.log.use_wandb:
            wandb.finish()
