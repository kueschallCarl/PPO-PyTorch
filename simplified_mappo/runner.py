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
        """
        try:
            logging.info("Starting evaluation...")
            eval_rewards = []
            position_history = {
                'agents': [],
                'landmarks': [],
                'distances': [],
                'rewards': [],
                'actions': []
            }
            
            # Create figure once before evaluation starts if visualizing
            if visualize:
                plt.ion()  # Turn on interactive mode
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                lines = []  # Store line objects for updating
                for agent_idx in range(self.num_agents):
                    line, = ax2.plot([], [], label=f'Agent {agent_idx}')
                    lines.append(line)
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Distance to Closest Landmark')
                ax2.legend()
                ax2.grid(True)
                plt.tight_layout()
            
            for episode in range(n_episodes):
                obs = self.env.reset()
                episode_reward = 0
                
                # Track positions and metrics only for first episode if visualizing
                is_tracking = visualize and episode == 0
                
                if is_tracking:
                    # Store initial positions
                    position_history['agents'].append(
                        [agent.state.p_pos.copy() for agent in self.world.agents]
                    )
                    position_history['landmarks'].append(
                        [l.state.p_pos.copy() for l in self.world.landmarks]
                    )
                    
                    # Calculate initial distances
                    distances = []
                    for agent in self.world.agents:
                        agent_distances = [
                            np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) 
                            for l in self.world.landmarks
                        ]
                        distances.append(agent_distances)
                    position_history['distances'].append(distances)
                
                for step in range(eval_episode_length):
                    # Get actions
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(np.stack(obs)).to(self.device)
                        actions, _ = policy.get_actions(obs_tensor, deterministic=True)
                    actions_np = actions.cpu().numpy()
                    
                    if is_tracking:
                        # Track actions
                        for i, action in enumerate(actions_np):
                            position_history['actions'].append({
                                f'agent_{i}': {
                                    'action': action.copy(),
                                    'position': self.world.agents[i].state.p_pos.copy(),
                                    'velocity': self.world.agents[i].state.p_vel.copy()
                                }
                            })
                    
                    # Execute actions
                    next_obs, rewards, dones, info = self.env.step(actions_np)
                    episode_reward += np.mean(rewards)
                    
                    if is_tracking:
                        # Track positions and distances after step
                        position_history['agents'].append(
                            [agent.state.p_pos.copy() for agent in self.world.agents]
                        )
                        position_history['landmarks'].append(
                            [l.state.p_pos.copy() for l in self.world.landmarks]
                        )
                        
                        # Calculate distances
                        distances = []
                        for agent in self.world.agents:
                            agent_distances = [
                                np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) 
                                for l in self.world.landmarks
                            ]
                            distances.append(agent_distances)
                        position_history['distances'].append(distances)
                        position_history['rewards'].append(rewards)
                        
                        # Update visualization
                        if visualize:
                            # Clear axes but keep figure
                            ax1.clear()
                            
                            # Update main visualization
                            render_env(self.world, ax=ax1)
                            ax1.set_xlim(-1.5, 1.5)
                            ax1.set_ylim(-1.5, 1.5)
                            ax1.grid(True)
                            
                            # Update distance plot
                            for agent_idx, line in enumerate(lines):
                                agent_distances = [d[agent_idx] for d in position_history['distances']]
                                min_distances = [min(d) for d in agent_distances]
                                line.set_data(range(len(min_distances)), min_distances)
                            
                            # Adjust distance plot limits
                            ax2.relim()
                            ax2.autoscale_view()
                            
                            # Update title with current step
                            fig.suptitle(f'Step {step}/{eval_episode_length}')
                            
                            # Refresh display
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            plt.pause(eval_delay)
                    
                    obs = next_obs
                    if all(dones):
                        break
                
                eval_rewards.append(episode_reward)
                
            # Close visualization
            if visualize:
                plt.close(fig)
                plt.ioff()
            
            mean_reward = np.mean(eval_rewards)
            return mean_reward, position_history if visualize else mean_reward
            
        except Exception as e:
            logging.error(f"Error in eval_policy: {str(e)}")
            logging.error(traceback.format_exc())
            if visualize:
                plt.close(fig)
                plt.ioff()
            raise