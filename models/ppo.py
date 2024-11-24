from torch.distributions import MultivariateNormal, Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.actor_critic import ActorCritic, CentralizedCritic
from utils.buffer import SharedReplayBuffer, AgentBatch
from config.config import Config
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np
from typing import Dict
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PPO:
    def __init__(self, state_dim, action_dim, num_agents, cfg: Config, writer: SummaryWriter = None):
        """Initialize PPO for MAPPO (Multi-Agent PPO) with centralized critic"""
        self.cfg = cfg
        self.writer = writer
        self.device = cfg.device
        self.total_updates = 0
        
        # Initialize hyperparameters
        self.eps_clip = cfg.ppo.eps_clip
        self.K_epochs = cfg.ppo.K_epochs
        self.value_loss_coef = cfg.ppo.value_loss_coef
        self.entropy_coef = cfg.ppo.entropy_coef
        self.max_grad_norm = cfg.ppo.max_grad_norm
        self.normalize_advantages = cfg.ppo.normalize_advantages
        self.has_continuous_action_space = cfg.env.has_continuous_action_space
        # Initialize actor-critic
        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            has_continuous_action_space=cfg.env.has_continuous_action_space,
            action_std_init=cfg.action.action_std
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': cfg.ppo.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': cfg.ppo.lr_critic}
        ])

    def select_action(self, state, actions, global_state):
        """Select action using the policy"""
        with torch.no_grad():
            # Convert to tensor if not already and add batch dimension if needed
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            if not isinstance(global_state, torch.Tensor):
                global_state = torch.FloatTensor(global_state).to(self.device)
            if global_state.dim() == 2:
                global_state = global_state.unsqueeze(0)
            
            if actions is not None:
                if not isinstance(actions, torch.Tensor):
                    actions = torch.FloatTensor(actions).to(self.device)
                if actions.dim() == 2:
                    actions = actions.unsqueeze(0)
            
            action, action_logprob, state_val = self.policy.act(
                state=state,
                global_state=global_state,
                actions=actions
            )
            
            print(f"Debug - select_action logprob shape: {action_logprob.shape}, value: {action_logprob.item()}")
            # Remove batch dimension and convert to numpy
            action = action.squeeze(0).cpu().numpy()
            action = np.clip(action, 0, 1)
            # Squeeze logprob to make it a scalar
            action_logprob = action_logprob.squeeze()
            return action, action_logprob

    def update(self, states, actions, rewards, next_states, dones, agent_idx, agent_batch, logprobs=None):
        """Update policy using MAPPO algorithm"""
        if self.writer is None:
            print("Warning: No writer available for logging!")
            return
        
        # Track key metrics during training
        running_metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy_loss': 0,
            'total_loss': 0,
            'clip_fraction': 0,
            'approx_kl': 0
        }

        # PPO update for K epochs
        for epoch in range(self.K_epochs):
            # Actor update (decentralized)
            action_mean = self.policy.actor(agent_batch.states)
            
            if self.has_continuous_action_space:
                action_var = self.policy.action_var.expand_as(action_mean)
                cov_mat = torch.diag_embed(action_var).to(self.device)
                dist = MultivariateNormal(action_mean, cov_mat)
                
                # Debug prints with proper tensor handling
                print(f"\nDebug - Agent {agent_idx}, Epoch {epoch}:")
                print(f"Action mean range: {action_mean.min().item():.3f} to {action_mean.max().item():.3f}")
                print(f"Action var range: {action_var.min().item():.3f} to {action_var.max().item():.3f}")
                
                action_logprobs = dist.log_prob(agent_batch.actions)
                dist_entropy = dist.entropy()
                ratios = torch.exp(action_logprobs - agent_batch.logprobs.detach())
                
                print(f"Raw action logprobs range: {action_logprobs.min().item():.3f} to {action_logprobs.max().item():.3f}")
                print(f"Old logprobs range: {agent_batch.logprobs.min().item():.3f} to {agent_batch.logprobs.max().item():.3f}")
                print(f"Ratio range: {ratios.min().item():.3f} to {ratios.max().item():.3f}")
                print(f"Entropy range: {dist_entropy.min().item():.3f} to {dist_entropy.max().item():.3f}")
            
            # Critic update (centralized)
            state_values = self.policy.critic(states, actions)
            
            # Policy loss (decentralized)
            surr1 = ratios * agent_batch.advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * agent_batch.advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (centralized)
            value_loss = F.mse_loss(state_values.squeeze(-1), agent_batch.returns)
            
            # Entropy loss
            entropy_loss = -dist_entropy.mean()

            # Total loss
            total_loss = (
                policy_loss +
                self.value_loss_coef * value_loss +
                self.entropy_coef * entropy_loss
            )

            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update metrics
            running_metrics['policy_loss'] += policy_loss.item()
            running_metrics['value_loss'] += value_loss.item()
            running_metrics['entropy_loss'] += entropy_loss.item()
            running_metrics['total_loss'] += total_loss.item()
            running_metrics['clip_fraction'] += ((ratios - 1.0).abs() > self.eps_clip).float().mean().item()
            running_metrics['approx_kl'] += 0.5 * ((agent_batch.logprobs - action_logprobs) ** 2).mean().item()

        # Average metrics over epochs
        for key in running_metrics:
            running_metrics[key] /= self.K_epochs

        # Log to tensorboard with simpler prefix structure
        prefix = f'Agent_{agent_idx}'
        
        # Training losses
        self.writer.add_scalar(f'{prefix}/policy_loss', running_metrics['policy_loss'], self.total_updates)
        self.writer.add_scalar(f'{prefix}/value_loss', running_metrics['value_loss'], self.total_updates)
        self.writer.add_scalar(f'{prefix}/entropy_loss', running_metrics['entropy_loss'], self.total_updates)
        self.writer.add_scalar(f'{prefix}/total_loss', running_metrics['total_loss'], self.total_updates)
        
        # PPO metrics
        self.writer.add_scalar(f'{prefix}/clip_fraction', running_metrics['clip_fraction'], self.total_updates)
        self.writer.add_scalar(f'{prefix}/approx_kl', running_metrics['approx_kl'], self.total_updates)
        
        # Learning rates
        self.writer.add_scalar(f'{prefix}/lr_actor', self.optimizer.param_groups[0]['lr'], self.total_updates)
        self.writer.add_scalar(f'{prefix}/lr_critic', self.optimizer.param_groups[1]['lr'], self.total_updates)

        # Force flush the writer
        self.writer.flush()

        # Debug prints
        print(f"\nAgent {agent_idx} Update {self.total_updates}:")
        print(f"Policy Loss: {running_metrics['policy_loss']:.6f}")
        print(f"Value Loss: {running_metrics['value_loss']:.6f}")
        print(f"Total Loss: {running_metrics['total_loss']:.6f}")

        self.total_updates += 1
        return running_metrics

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))