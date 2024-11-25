import torch
import torch.nn as nn
import numpy as np

class IPPOPolicy(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64, layer_N=2):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_space, hidden_size),
            nn.ReLU(),
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (layer_N-1)),
            nn.Linear(hidden_size, action_space),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_space, hidden_size),
            nn.ReLU(),
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (layer_N-1)),
            nn.Linear(hidden_size, 1)
        )
        
        # Action std for exploration
        self.log_std = nn.Parameter(torch.zeros(action_space))
        
    def get_actions(self, obs, deterministic=False):
        action_mean = self.actor(obs)
        
        if deterministic:
            return action_mean, None
            
        std = self.log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        
        return actions, action_log_probs
        
    def evaluate_actions(self, obs, actions):
        action_mean = self.actor(obs)
        std = self.log_std.exp()
        
        dist = torch.distributions.Normal(action_mean, std)
        action_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().mean()
        
        values = self.critic(obs)
        
        return action_log_probs, values, dist_entropy
