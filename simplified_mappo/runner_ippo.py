import numpy as np
import torch
from gym.spaces import Box
import gym
import time
from envs import MPEEnv
import logging
import traceback
from utils.visualization import render_env
import matplotlib.pyplot as plt

class RunnerIPPO:
    def __init__(self, env_name, num_agents, seed=1, device='cpu'):
        self.env = MPEEnv(env_name, num_agents)
        self.num_agents = num_agents
        self.device = device
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        
        # Get environment specs
        self.obs_space = self.env.observation_space[0].shape[0]
        self.action_space = self.env.action_space[0].shape[0]
        
        logging.info(f"Initialized IPPO Runner with {num_agents} agents on device: {device}")
        logging.info(f"Observation space: {self.obs_space}, Action space: {self.action_space}")
    
    def collect_episodes(self, policies, buffers, episode_length):
        try:
            obs = self.env.reset()
            episode_rewards = []
            
            # Log initial observation shape
            logging.debug(f"Initial observation shape: {np.array(obs).shape}")
            
            for step in range(episode_length):
                actions = []
                action_log_probs = []
                values = []
                
                for agent_id, policy in enumerate(policies):
                    obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(self.device)
                    logging.debug(f"Step {step}, Agent {agent_id} - obs_tensor shape: {obs_tensor.shape}")
                    
                    with torch.no_grad():
                        action, action_log_prob = policy.get_actions(obs_tensor)
                        value = policy.critic(obs_tensor)
                        
                        logging.debug(f"Step {step}, Agent {agent_id} shapes:")
                        logging.debug(f"  - action: {action.shape}")
                        logging.debug(f"  - action_log_prob: {action_log_prob.shape if action_log_prob is not None else 'None'}")
                        logging.debug(f"  - value: {value.shape}")
                    
                    actions.append(action.squeeze(0).cpu().numpy())
                    action_log_probs.append(action_log_prob.cpu().numpy())
                    values.append(value.cpu().numpy())
                
                # Log shapes after processing
                logging.debug(f"Step {step} - Processed shapes:")
                logging.debug(f"  - actions: {np.array(actions).shape}")
                logging.debug(f"  - action_log_probs: {np.array(action_log_probs).shape}")
                logging.debug(f"  - values: {np.array(values).shape}")
                
                next_obs, rewards, dones, info = self.env.step(actions)
                
                # Log shapes before buffer insertion
                for agent_id in range(self.num_agents):
                    logging.debug(f"Step {step}, Agent {agent_id} - Buffer insertion shapes:")
                    logging.debug(f"  - obs: {np.array([obs[agent_id]]).shape}")
                    logging.debug(f"  - actions: {np.array([actions[agent_id]]).shape}")
                    logging.debug(f"  - rewards: {np.array([[rewards[agent_id]]]).shape}")
                    logging.debug(f"  - value_preds: {np.array(values[agent_id]).shape}")
                    logging.debug(f"  - action_log_probs: {np.array([action_log_probs[agent_id]]).shape}")
                    logging.debug(f"  - masks: {np.array([[0.0] if dones[agent_id] else [1.0]]).shape}")
                    
                    try:
                        # Reshape the data to match buffer dimensions (num_agents=1, feature_dim)
                        obs_reshaped = np.array([[obs[agent_id]]])  # Shape: (1, 1, obs_space)
                        actions_reshaped = np.array([[actions[agent_id]]])  # Shape: (1, 1, action_space)
                        rewards_reshaped = np.array([[[rewards[agent_id]]]])  # Shape: (1, 1, 1)
                        value_preds_reshaped = np.array([[[values[agent_id].item()]]])  # Shape: (1, 1, 1)
                        action_log_probs_reshaped = np.array([[[action_log_probs[agent_id].item()]]])  # Shape: (1, 1, 1)
                        masks_reshaped = np.array([[[0.0] if dones[agent_id] else [1.0]]])  # Shape: (1, 1, 1)
                        
                        buffers[agent_id].insert(
                            obs=obs_reshaped[0],  # Remove the extra dimension we added
                            actions=actions_reshaped[0],
                            rewards=rewards_reshaped[0],
                            value_preds=value_preds_reshaped[0],
                            action_log_probs=action_log_probs_reshaped[0],
                            masks=masks_reshaped[0]
                        )
                    except Exception as e:
                        logging.error(f"Buffer insertion failed for agent {agent_id}")
                        logging.error(f"values[agent_id] content: {values[agent_id]}")
                        logging.error(f"values[agent_id] type: {type(values[agent_id])}")
                        logging.error(f"values[agent_id] shape: {np.array(values[agent_id]).shape}")
                        logging.error(f"Reshaped value_preds shape: {value_preds_reshaped.shape}")
                        raise
                
                obs = next_obs
                episode_rewards.append(rewards)
                
                if all(dones):
                    obs = self.env.reset()
            
            # Compute returns for each agent
            for agent_id, policy in enumerate(policies):
                with torch.no_grad():
                    next_obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(self.device)
                    next_value = policy.critic(next_obs_tensor).cpu().numpy()
                    logging.debug(f"Agent {agent_id} - next_value shape: {next_value.shape}")
                    buffers[agent_id].compute_returns(next_value, gamma=0.99, gae_lambda=0.95)
            
            return np.mean(episode_rewards)
            
        except Exception as e:
            logging.error(f"Error in collect_episodes: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def eval_policy(self, policies, n_episodes=5, visualize=False, eval_delay=0.05):
        try:
            eval_rewards = []
            fig = None
            position_history = {'distances': []} if visualize else None
            
            if visualize:
                plt.ion()
                fig, ax = plt.subplots(figsize=(8, 8))
            
            for episode in range(n_episodes):
                obs = self.env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                max_steps = 100
                
                # Initialize agent trails
                agent_trails = {f'agent_{i}': [] for i in range(self.num_agents)} if visualize else None
                
                while not done and step_count < max_steps:
                    step_count += 1
                    actions = []
                    
                    for agent_id, policy in enumerate(policies):
                        obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action, _ = policy.get_actions(obs_tensor, deterministic=True)
                        actions.append(action.squeeze(0).cpu().numpy())
                    
                    next_obs, rewards, dones, _ = self.env.step(actions)
                    episode_reward += np.mean(rewards)
                    done = all(dones)
                    
                    if visualize and episode == 0:
                        ax.clear()
                        distances, agent_trails = render_env(self.env, ax=ax, agent_trails=agent_trails)
                        if distances is not None:
                            position_history['distances'].append(distances)
                        plt.draw()
                        plt.pause(eval_delay)
                    
                    obs = next_obs
                
                eval_rewards.append(episode_reward)
            
            if visualize and fig is not None:
                plt.ioff()
                plt.close(fig)
            
            mean_reward = np.mean(eval_rewards)
            
            # Return position_history for visualization metrics
            return mean_reward if not visualize else (mean_reward, position_history)
            
        except Exception as e:
            logging.error(f"Error in eval_policy: {str(e)}")
            logging.error(traceback.format_exc())
            if visualize and fig is not None:
                plt.close(fig)
                plt.ioff()
            raise
