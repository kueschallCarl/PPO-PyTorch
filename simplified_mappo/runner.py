import numpy as np
import torch
from gym.spaces import Box
import gym
import time
from envs import MPEEnv
import logging
import traceback

class Runner:
    def __init__(self, env_name, num_agents, seed=1, device='cpu'):
        self.env = MPEEnv("simple_spread", num_agents)
        self.num_agents = num_agents
        self.device = device
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        
        # Get environment specs
        self.obs_space = self.env.observation_space[0].shape[0]
        self.action_space = self.env.action_space[0].shape[0]
        
        logging.info(f"Initialized Runner with {num_agents} agents on device: {device}")
        logging.info(f"Observation space: {self.obs_space}, Action space: {self.action_space}")
        
    def collect_episodes(self, policy, buffer, episode_length):
        try:
            collection_start = time.time()
            obs = self.env.reset()
            episode_rewards = []
            episode_steps = 0
            num_resets = 0
            
            for step in range(episode_length):
                # Convert observations to tensor and move to correct device
                obs_tensor = torch.FloatTensor(np.stack(obs)).to(self.device)
                
                # Get actions
                with torch.no_grad():
                    actions, action_log_probs = policy.get_actions(obs_tensor)
                    values = policy.critic(obs_tensor)
                
                # Execute actions
                actions_np = actions.cpu().numpy()
                next_obs, rewards, dones, info = self.env.step(actions_np)
                
                # Create masks for done episodes
                masks = [[0.0] if done else [1.0] for done in dones]
                
                # Reshape rewards to match buffer expectations (num_agents, 1)
                rewards_reshaped = np.array(rewards).reshape(self.num_agents, 1)
                
                # Store transition
                buffer.insert(obs, actions_np, rewards_reshaped, 
                            values.cpu().numpy(),
                            action_log_probs.cpu().numpy(),
                            masks)
                
                obs = next_obs
                episode_rewards.append(rewards)
                
                if all(dones):
                    obs = self.env.reset()
                    num_resets += 1
                    logging.debug(f"Episode reset at step {step}, total resets: {num_resets}")
                
                episode_steps = step + 1
            
            # Compute returns
            with torch.no_grad():
                next_obs_tensor = torch.FloatTensor(np.stack(obs)).to(self.device)
                next_values = policy.critic(next_obs_tensor).cpu().numpy()
            buffer.compute_returns(next_values, gamma=0.99, gae_lambda=0.95)
            
            collection_time = time.time() - collection_start
            mean_reward = np.mean(episode_rewards)
            logging.debug(f"Collected {episode_steps} steps in {collection_time:.2f}s. "
                         f"Mean reward: {mean_reward:.3f}, Complete episodes: {num_resets}")
            
            return mean_reward
            
        except Exception as e:
            logging.error(f"Error in collect_episodes: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def eval_policy(self, policy, n_episodes=5, eval_episode_length=25):
        try:
            logging.info("Starting evaluation...")
            eval_start = time.time()
            eval_rewards = []
            total_steps = 0
            
            for episode in range(n_episodes):
                logging.debug(f"Starting evaluation episode {episode + 1}/{n_episodes}")
                obs = self.env.reset()
                episode_reward = 0
                
                # Run for fixed number of steps instead of waiting for done
                for step in range(eval_episode_length):
                    try:
                        with torch.no_grad():
                            obs_tensor = torch.FloatTensor(np.stack(obs)).to(self.device)
                            actions, _ = policy.get_actions(obs_tensor, deterministic=True)
                        
                        obs, rewards, _, _ = self.env.step(actions.cpu().numpy())
                        episode_reward += np.mean(rewards)
                        total_steps += 1
                        
                    except Exception as e:
                        logging.error(f"Error in evaluation step: {str(e)}")
                        logging.error(traceback.format_exc())
                        raise
                
                eval_rewards.append(episode_reward)
                logging.debug(f"Eval episode {episode + 1}/{n_episodes} completed: "
                            f"Reward: {episode_reward:.3f}")
            
            mean_reward = np.mean(eval_rewards)
            eval_time = time.time() - eval_start
            logging.info(f"Evaluation completed in {eval_time:.2f}s. "
                        f"Mean reward: {mean_reward:.3f}, "
                        f"Total steps: {total_steps}")
            
            return mean_reward
            
        except Exception as e:
            logging.error(f"Error in eval_policy: {str(e)}")
            logging.error(traceback.format_exc())
            raise