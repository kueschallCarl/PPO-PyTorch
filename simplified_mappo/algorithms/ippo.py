import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

class IPPO:
    def __init__(self, policy, cfg):
        self.policy = policy
        self.cfg = cfg
        
        self.actor_optimizer = optim.Adam(
            self.policy.actor.parameters(), 
            lr=cfg.training.lr_actor
        )
        self.critic_optimizer = optim.Adam(
            self.policy.critic.parameters(),
            lr=cfg.training.lr_critic
        )
        
    def update(self, sample: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """Update policy using the collected samples"""
        
        # Get tensors from sample
        obs = sample['obs']
        actions = sample['actions']
        returns = sample['returns']
        advantages = sample['advantages']
        old_action_log_probs = sample['action_log_probs']
        
        # Evaluate actions
        action_log_probs, values, dist_entropy = self.policy.evaluate_actions(obs, actions)
        
        # Value loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Policy loss
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.training.clip_ratio,
                           1.0 + self.cfg.training.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Total loss
        loss = (policy_loss * self.cfg.training.policy_loss_coef + 
                value_loss * self.cfg.training.value_loss_coef - 
                dist_entropy * self.cfg.training.entropy_coef)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.cfg.training.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(),
                self.cfg.training.max_grad_norm
            )
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self.cfg.training.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(),
                self.cfg.training.max_grad_norm
            )
        self.critic_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': dist_entropy.item()
        }
