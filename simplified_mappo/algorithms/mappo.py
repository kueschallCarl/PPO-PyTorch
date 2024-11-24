import torch
import torch.nn as nn
import torch.optim as optim

class MAPPO:
    def __init__(self, policy, lr=3e-4, clip_param=0.2, value_loss_coef=1.0, 
                 entropy_coef=0.01, max_grad_norm=0.5):
        self.policy = policy
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
    def update(self, sample):
        obs_batch = sample['obs']
        actions_batch = sample['actions']
        value_preds_batch = sample['value_preds']
        return_batch = sample['returns']
        old_action_log_probs_batch = sample['action_log_probs']
        adv_targ = sample['advantages']
        
        # Normalize advantages
        adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)

        # Policy loss
        action_log_probs, values, dist_entropy = self.policy.evaluate_actions(obs_batch, actions_batch)
        
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = 0.5 * (return_batch - values).pow(2).mean()

        # Total loss
        loss = (action_loss + self.value_loss_coef * value_loss - 
                self.entropy_coef * dist_entropy)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'policy_loss': action_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': dist_entropy.item()
        } 