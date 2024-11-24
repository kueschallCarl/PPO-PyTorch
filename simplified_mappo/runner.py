import numpy as np
import torch
from gym.spaces import Box
import gym
import time
from envs import MPEEnv
import logging
import traceback
import matplotlib.pyplot as plt
from utils.visualization import render_env

class Runner:
    def __init__(self, env_name, num_agents, seed=1, device='cpu'):
        self.env = MPEEnv("simple_spread", num_agents)
        self.num_agents = num_agents
        self.device = device
        self.world = self.env.world  # Store reference to world for visualization
        
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
            
            return mean_reward
            
        except Exception as e:
            logging.error(f"Error in collect_episodes: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def eval_policy(self, policy, n_episodes=5, eval_episode_length=25, visualize=False, eval_delay=0.25):
        """
        Evaluate policy for multiple episodes
        
        Args:
            policy: Policy to evaluate
            n_episodes: Number of episodes to evaluate
            eval_episode_length: Maximum length of each episode
            visualize: Whether to visualize the first evaluation episode
            eval_delay: Delay between steps during visualization
        """
        try:
            logging.info("Starting evaluation...")
            eval_start = time.time()
            eval_rewards = []
            total_steps = 0
            
            for episode in range(n_episodes):
                logging.debug(f"Starting evaluation episode {episode + 1}/{n_episodes}")
                obs = self.env.reset()
                episode_reward = 0
                
                # Create figure for visualization if needed
                if visualize and episode == 0:  # Only visualize first episode
                    plt.figure(figsize=(8, 8))
                
                for step in range(eval_episode_length):
                    try:
                        # Store previous positions for interpolation if visualizing
                        if visualize and episode == 0:
                            prev_positions = np.array([agent.state.p_pos.copy() for agent in self.world.agents])
                            prev_velocities = np.array([agent.state.p_vel.copy() for agent in self.world.agents])
                        
                        # Get actions
                        with torch.no_grad():
                            obs_tensor = torch.FloatTensor(np.stack(obs)).to(self.device)
                            actions, _ = policy.get_actions(obs_tensor, deterministic=True)
                        actions_np = actions.cpu().numpy()
                        
                        # Step environment
                        obs, rewards, dones, _ = self.env.step(actions_np)
                        episode_reward += np.mean(rewards)
                        total_steps += 1
                        
                        # Visualize if needed
                        if visualize and episode == 0:
                            # Get new positions after step
                            new_positions = np.array([agent.state.p_pos.copy() for agent in self.world.agents])
                            new_velocities = np.array([agent.state.p_vel.copy() for agent in self.world.agents])
                            
                            # Interpolate for smooth visualization
                            n_interp = 10
                            for i in range(n_interp):
                                t = i / n_interp
                                # Linearly interpolate positions and velocities
                                interp_positions = prev_positions * (1 - t) + new_positions * t
                                interp_velocities = prev_velocities * (1 - t) + new_velocities * t
                                
                                # Update agent states for rendering
                                for agent_idx, agent in enumerate(self.world.agents):
                                    agent.state.p_pos = interp_positions[agent_idx]
                                    agent.state.p_vel = interp_velocities[agent_idx]
                                
                                render_env(self.world)
                                plt.pause(eval_delay / n_interp)
                            
                            # Restore final positions and velocities
                            for agent_idx, agent in enumerate(self.world.agents):
                                agent.state.p_pos = new_positions[agent_idx]
                                agent.state.p_vel = new_velocities[agent_idx]
                        
                        if all(dones):
                            break
                            
                    except Exception as e:
                        logging.error(f"Error in evaluation step: {str(e)}")
                        logging.error(traceback.format_exc())
                        raise
                
                # Close visualization for this episode
                if visualize and episode == 0:
                    plt.close()
                
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