from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models.ppo import PPO
from config.config import Config
from utils.buffer import SharedReplayBuffer

class MAPPOTrainer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        cfg: Config,
        writer: SummaryWriter
    ):
        self.cfg = cfg
        self.writer = writer
        self.num_agents = num_agents
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create shared replay buffer for all agents
        self.buffer = SharedReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            buffer_size=int(cfg.env.max_ep_len * cfg.ppo.update_timestep),
            device=self.device
        )

        # Initialize agents (actors) and shared critic
        self.agents = [
            PPO(
                state_dim=state_dim,
                action_dim=action_dim,
                cfg=cfg,
                writer=writer,
                critic_type='centralized',
                num_agents=num_agents
            ) for _ in range(num_agents)
        ]

        self.total_steps = 0
        self.update_count = 0
        
        # Add running averages for more stable logging
        self.running_losses = {
            agent_idx: {
                'policy_loss': [],
                'value_loss': [],
                'entropy_loss': [],
                'total_loss': [],
                'approx_kl': [],
                'clip_fraction': [],
                'explained_variance': []
            } for agent_idx in range(num_agents)
        }
        self.log_window = 100  # Window size for running average

    def select_actions(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False) -> Dict[str, np.ndarray]:
        """Select actions for all agents using their respective policies"""
        actions = {}
        with torch.no_grad():
            # Create global state tensor with proper ordering
            global_states = []
            for agent_id in sorted(obs_dict.keys()):
                obs = obs_dict[agent_id]
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                global_states.append(obs_tensor)
            global_states = torch.stack(global_states, dim=0)  # [num_agents, state_dim]
            
            # Initialize actions tensor
            prev_actions = torch.zeros(
                self.num_agents, 
                self.agents[0].policy.actor.network[-1].out_features
            ).to(self.device)
            
            # Select actions for each agent
            for agent_id in sorted(obs_dict.keys()):
                agent_idx = int(agent_id.split('_')[1])
                obs_tensor = torch.FloatTensor(obs_dict[agent_id]).to(self.device)
                
                # Select action using the agent's policy
                action = self.agents[agent_idx].select_action(
                    state=obs_tensor,
                    actions=prev_actions,
                    global_state=global_states,
                    deterministic=deterministic
                )
                
                actions[agent_id] = action
                
                # Update previous actions tensor
                if isinstance(action, np.ndarray):
                    prev_actions[agent_idx] = torch.FloatTensor(action).to(self.device)
                else:
                    prev_actions[agent_idx] = torch.tensor([action]).float().to(self.device)
                    
        return actions

    def update(self):
        """Update all agent policies"""
        states, actions, rewards, next_states, dones = self.buffer.get_all()
        
        # Update total steps
        self.total_steps += len(rewards)
        
        # Update each agent
        for agent_idx, agent in enumerate(self.agents):
            losses = agent.update_mappo(
                states, actions, rewards, next_states, dones, agent_idx
            )
            
            # Log all metrics directly without running averages
            for key, value in losses.items():
                self.writer.add_scalar(
                    f'Agent_{agent_idx}/Metrics/{key}',
                    value,
                    self.total_steps
                )

    def log_pre_update_stats(self, states, actions, rewards):
        """Log statistics before policy update"""
        self.writer.add_scalar('Training/mean_reward', rewards.mean().item(), self.total_steps)
        self.writer.add_scalar('Training/reward_std', rewards.std().item(), self.total_steps)
        
        # Log state and action statistics
        self.writer.add_scalar('State/mean', states.mean().item(), self.total_steps)
        self.writer.add_scalar('State/std', states.std().item(), self.total_steps)
        self.writer.add_scalar('Action/mean', actions.mean().item(), self.total_steps)
        self.writer.add_scalar('Action/std', actions.std().item(), self.total_steps)

    def log_post_update_stats(self):
        """Log statistics after policy update"""
        # Log policy parameters for each agent
        for agent_idx, agent in enumerate(self.agents):
            for name, param in agent.policy.named_parameters():
                self.writer.add_histogram(
                    f'Agent_{agent_idx}/Parameters/{name}', 
                    param.data, 
                    self.total_steps
                )

    def save(self, path: str):
        """Save all agent models"""
        for idx, agent in enumerate(self.agents):
            agent_path = path.replace('.pth', f'_agent{idx}.pth')
            agent.save(agent_path)

    def load(self, path: str):
        """Load all agent models"""
        for idx, agent in enumerate(self.agents):
            agent_path = path.replace('.pth', f'_agent{idx}.pth')
            agent.load(agent_path)

    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float):
        """Decay action standard deviation for all agents"""
        if self.cfg.env.has_continuous_action_space:
            for agent in self.agents:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

    def update_total_steps(self, steps: int):
        """Update total environment steps"""
        self.total_steps += steps 

    def log_episode_stats(self, episode_reward, episode_length):
        """Log episode-level statistics"""
        self.writer.add_scalar('Training/episode_reward', episode_reward, self.total_steps)
        self.writer.add_scalar('Training/episode_length', episode_length, self.total_steps)