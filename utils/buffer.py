import torch
import numpy as np
from typing import Tuple, List, Dict

class SharedReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        buffer_size: int,
        device: torch.device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.device = device
        
        # Initialize buffers for all agents
        self.states = torch.zeros((buffer_size, num_agents, state_dim)).to(device)
        self.actions = torch.zeros((buffer_size, num_agents, action_dim)).to(device)
        self.rewards = torch.zeros((buffer_size, num_agents)).to(device)
        self.next_states = torch.zeros((buffer_size, num_agents, state_dim)).to(device)
        self.dones = torch.zeros((buffer_size, num_agents)).to(device)
        
        # Add PPO-specific buffers for each agent
        self.logprobs = torch.zeros((buffer_size, num_agents)).to(device)
        self.values = torch.zeros((buffer_size, num_agents)).to(device)
        self.returns = torch.zeros((buffer_size, num_agents)).to(device)
        self.advantages = torch.zeros((buffer_size, num_agents)).to(device)
        
        self.ptr = 0
        self.size = 0

    def add(
        self,
        states: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_states: Dict[str, np.ndarray],
        dones: Dict[str, bool],
        logprobs: Dict[str, torch.Tensor] = None,
        values: Dict[str, torch.Tensor] = None
    ):
        # Convert dictionary data to tensors and store
        for agent_id in states.keys():
            agent_idx = int(agent_id.split('_')[1])
            
            self.states[self.ptr, agent_idx] = torch.FloatTensor(states[agent_id]).to(self.device)
            self.actions[self.ptr, agent_idx] = torch.FloatTensor(actions[agent_id]).to(self.device)
            self.rewards[self.ptr, agent_idx] = torch.FloatTensor([rewards[agent_id]]).to(self.device)
            self.next_states[self.ptr, agent_idx] = torch.FloatTensor(next_states[agent_id]).to(self.device)
            self.dones[self.ptr, agent_idx] = torch.FloatTensor([dones[agent_id]]).to(self.device)
            
            # Store PPO-specific data if provided
            if logprobs is not None:
                self.logprobs[self.ptr, agent_idx] = logprobs[agent_id].to(self.device)
            if values is not None:
                self.values[self.ptr, agent_idx] = values[agent_id].to(self.device)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute returns and advantages for all agents using GAE"""
        advantages = torch.zeros_like(self.rewards[:self.size])
        last_gae_lam = torch.zeros((self.num_agents,)).to(self.device)
        
        # Get actual values and next values
        values = self.values[:self.size]
        next_values = torch.cat([values[1:], last_values.unsqueeze(0)], dim=0)
        
        # Compute GAE for each timestep
        for t in reversed(range(self.size)):
            delta = (
                self.rewards[t] + 
                gamma * next_values[t] * (1 - self.dones[t]) - 
                values[t]
            )
            last_gae_lam = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae_lam
            advantages[t] = last_gae_lam
            
        # Compute returns
        self.returns[:self.size] = advantages + values
        self.advantages[:self.size] = advantages

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all stored transitions"""
        return (
            self.states[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.next_states[:self.size],
            self.dones[:self.size]
        )

    def get_agent_data(self, agent_idx: int) -> Tuple[torch.Tensor, ...]:
        """Get data specific to one agent for PPO updates"""
        return (
            self.states[:self.size, agent_idx],
            self.actions[:self.size, agent_idx],
            self.logprobs[:self.size, agent_idx],
            self.returns[:self.size, agent_idx],
            self.advantages[:self.size, agent_idx],
            self.values[:self.size, agent_idx]
        )

    def clear(self):
        """Reset buffer"""
        self.ptr = 0
        self.size = 0 