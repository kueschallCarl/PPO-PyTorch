import numpy as np
import torch

class SharedReplayBuffer:
    def __init__(self, num_agents, obs_space, act_space, size, device):
        self.device = device
        self.size = size
        self.num_agents = num_agents
        
        # Core buffers
        self.observations = np.zeros((size, num_agents, obs_space), dtype=np.float32)
        self.actions = np.zeros((size, num_agents, act_space), dtype=np.float32)
        self.rewards = np.zeros((size, num_agents, 1), dtype=np.float32)
        self.value_preds = np.zeros((size, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((size, num_agents, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((size, num_agents, 1), dtype=np.float32)
        self.advantages = np.zeros((size, num_agents, 1), dtype=np.float32)
        self.masks = np.ones((size, num_agents, 1), dtype=np.float32)
        
        self.step = 0

    def insert(self, obs, actions, rewards, value_preds, action_log_probs, masks):
        self.observations[self.step] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.masks[self.step] = masks.copy()
        
        self.step = (self.step + 1) % self.size

    def compute_returns(self, next_value, gamma, gae_lambda):
        gae = 0
        for step in reversed(range(self.step)):
            delta = self.rewards[step] + gamma * next_value * self.masks[step] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step] * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.value_preds[step]
            next_value = self.value_preds[step]

    def get_samples(self, batch_size):
        indices = np.random.permutation(self.step)
        for start_idx in range(0, self.step, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield {
                'obs': torch.FloatTensor(self.observations[batch_indices]).to(self.device),
                'actions': torch.FloatTensor(self.actions[batch_indices]).to(self.device),
                'value_preds': torch.FloatTensor(self.value_preds[batch_indices]).to(self.device),
                'returns': torch.FloatTensor(self.returns[batch_indices]).to(self.device),
                'action_log_probs': torch.FloatTensor(self.action_log_probs[batch_indices]).to(self.device),
                'advantages': torch.FloatTensor(self.advantages[batch_indices]).to(self.device)
            } 