import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np

class MAPPO:
    def __init__(self, policy, cfg):
        self.policy = policy
        self.cfg = cfg
        
        self.clip_param = cfg.training.clip_ratio
        self.value_loss_coef = cfg.training.value_loss_coef
        self.entropy_coef = cfg.training.entropy_coef
        self.max_grad_norm = cfg.training.max_grad_norm
        
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': cfg.training.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': cfg.training.lr_critic}
        ])
        
        # Initialize step counters
        self.update_count = 0
        self.last_log_step = 0

    @staticmethod
    def safe_std(tensor):
        if tensor.numel() <= 1:
            return torch.tensor(0.0)
        return tensor.std(unbiased=False)
        
    def update(self, sample, global_step):
        # Update step counter
        self.update_count += 1
        
        obs_batch = sample['obs']
        actions_batch = sample['actions']
        value_preds_batch = sample['value_preds']
        return_batch = sample['returns']
        old_action_log_probs_batch = sample['action_log_probs']
        advantages = sample['advantages']
        
        # Normalize advantages
        if self.cfg.training.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Policy evaluation
        action_log_probs, values, dist_entropy = self.policy.evaluate_actions(obs_batch, actions_batch)
        
        # Calculate ratios and surrogate losses
        ratios = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss with optional clipping
        if self.cfg.training.use_value_clipping:
            value_pred_clipped = value_preds_batch + torch.clamp(
                values - value_preds_batch,
                -self.clip_param * value_preds_batch.abs(),
                self.clip_param * value_preds_batch.abs()
            )
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        # Value function regularization
        value_reg_loss = 0.01 * values.pow(2).mean()
        value_loss += value_reg_loss

        # Total loss
        loss = (
            policy_loss * self.cfg.training.policy_loss_coef + 
            value_loss * self.value_loss_coef - 
            self.entropy_coef * dist_entropy
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Get gradient norms before clipping
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), float('inf')).item()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.critic.parameters(), float('inf')).item()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm * 0.5)
        
        self.optimizer.step()

        # Log to wandb if enabled
        if self.cfg.log.use_wandb and global_step > self.last_log_step:
            wandb_logs = {
                # Loss metrics
                "losses/total_loss": loss.item(),
                "losses/value_loss": value_loss.item(),
                "losses/policy_loss": policy_loss.item(),
                "losses/entropy_loss": (-self.entropy_coef * dist_entropy).item(),
                "losses/value_reg_loss": value_reg_loss.item(),
                
                # Policy metrics
                "policy/mean_ratio": ratios.mean().item(),
                "policy/ratio_std": ratios.std().item(),
                "policy/ratio_max": ratios.max().item(),
                "policy/ratio_min": ratios.min().item(),
                "policy/entropy": dist_entropy.item(),
                "policy/mean_logprob": action_log_probs.mean().item(),
                "policy/logprob_std": action_log_probs.std().item(),
                
                # Value metrics
                "values/mean_value": values.mean().item(),
                "values/value_std": values.std().item(),
                "values/value_max": values.max().item(),
                "values/value_min": values.min().item(),
                "values/mean_return": return_batch.mean().item(),
                "values/return_std": return_batch.std().item(),
                
                # Advantage metrics
                "advantages/mean": advantages.mean().item(),
                "advantages/std": advantages.std().item(),
                "advantages/max": advantages.max().item(),
                "advantages/min": advantages.min().item(),
                
                # Gradient metrics
                "gradients/actor_grad_norm": actor_grad_norm,
                "gradients/critic_grad_norm": critic_grad_norm,
            }
            
            # Add clipping metrics if enabled
            if self.cfg.training.use_value_clipping:
                wandb_logs.update({
                    "values/clipped_fraction": (value_losses_clipped < value_losses).float().mean().item(),
                    "values/clipping_threshold": self.clip_param
                })
            
            # Add parameter statistics
            for name, param in self.policy.named_parameters():
                wandb_logs.update({
                    f"parameters/{name}_mean": param.data.mean().item(),
                    f"parameters/{name}_std": self.safe_std(param.data).item(),
                    f"parameters/{name}_grad_mean": param.grad.mean().item() if param.grad is not None else 0,
                    f"parameters/{name}_grad_std": self.safe_std(param.grad).item() if param.grad is not None else 0,
                })
            
            wandb.log(wandb_logs, step=global_step)
            self.last_log_step = global_step

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': dist_entropy.item(),
            'total_loss': loss.item(),
            'approx_kl': (old_action_log_probs_batch - action_log_probs).mean().item()
        } 