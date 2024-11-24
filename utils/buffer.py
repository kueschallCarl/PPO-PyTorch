import torch
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

@dataclass
class AgentBatch:
    """Data structure to hold agent-specific batch data"""
    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor = None
    returns: torch.Tensor = None

class SharedReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        buffer_size: int,
        device: torch.device,
        writer: SummaryWriter = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.device = device
        self.total_updates = 0
        
        # Initialize buffers with proper shapes
        self.states = torch.zeros((buffer_size, num_agents, state_dim)).to(device)
        self.actions = torch.zeros((buffer_size, num_agents, action_dim)).to(device)
        self.rewards = torch.zeros((buffer_size, num_agents)).to(device)
        self.next_states = torch.zeros((buffer_size, num_agents, state_dim)).to(device)
        self.dones = torch.zeros((buffer_size, num_agents)).to(device)
        
        # PPO-specific buffers
        self.logprobs = torch.zeros((buffer_size, num_agents)).to(device)
        self.values = torch.zeros((buffer_size, num_agents)).to(device)
        self.returns = torch.zeros((buffer_size, num_agents)).to(device)
        self.advantages = torch.zeros((buffer_size, num_agents)).to(device)
        
        self.ptr = 0
        self.size = 0
        
        # Add shape assertions
        self._verify_shapes()

    def _verify_shapes(self):
        """Verify that all tensors have correct shapes"""
        assert self.states.shape == (self.buffer_size, self.num_agents, self.state_dim)
        assert self.actions.shape == (self.buffer_size, self.num_agents, self.action_dim)
        assert self.rewards.shape == (self.buffer_size, self.num_agents)
        assert self.next_states.shape == (self.buffer_size, self.num_agents, self.state_dim)
        assert self.dones.shape == (self.buffer_size, self.num_agents)
        assert self.logprobs.shape == (self.buffer_size, self.num_agents)
        assert self.values.shape == (self.buffer_size, self.num_agents)
        assert self.returns.shape == (self.buffer_size, self.num_agents)
        assert self.advantages.shape == (self.buffer_size, self.num_agents)

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
        """Add transition to buffer with shape verification"""
        # Check if buffer is full
        if self.size >= self.buffer_size:
            return False
            
        # Verify input dictionary sizes
        assert len(states) == self.num_agents, f"Expected {self.num_agents} agents, got {len(states)}"
        
        # Convert and store data for each agent
        for agent_id in sorted(states.keys()):  # Sort to ensure consistent ordering
            agent_idx = int(agent_id.split('_')[1])
            
            # Convert to tensors and verify shapes
            state = torch.FloatTensor(states[agent_id]).to(self.device)
            action = torch.FloatTensor(actions[agent_id]).to(self.device)
            reward = torch.FloatTensor([rewards[agent_id]]).to(self.device)
            next_state = torch.FloatTensor(next_states[agent_id]).to(self.device)
            done = torch.FloatTensor([dones[agent_id]]).to(self.device)
            
            # Verify individual tensor shapes
            assert state.shape == (self.state_dim,), f"Invalid state shape: {state.shape}"
            assert action.shape == (self.action_dim,), f"Invalid action shape: {action.shape}"
            
            # Store in buffer
            self.states[self.ptr, agent_idx] = state
            self.actions[self.ptr, agent_idx] = action
            self.rewards[self.ptr, agent_idx] = reward
            self.next_states[self.ptr, agent_idx] = next_state
            self.dones[self.ptr, agent_idx] = done
            
            # Store PPO-specific data if provided
            if logprobs is not None:
                logprob = logprobs[agent_id].to(self.device)
                logprob = logprob.squeeze()
                self.logprobs[self.ptr, agent_idx] = logprob
                
            if values is not None:
                value = values[agent_id].to(self.device)
                value = value.squeeze()
                self.values[self.ptr, agent_idx] = value

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        return True

    def compute_gae_and_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute GAE and returns for all agents
        Args:
            last_values: tensor of shape [num_agents] containing value estimates for last states
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        """
        # Ensure last_values has correct shape
        assert last_values.shape == (self.num_agents,), f"Expected last_values shape ({self.num_agents},), got {last_values.shape}"
        
        # Initialize advantages and returns tensors
        self.advantages = torch.zeros_like(self.rewards[:self.size])
        self.returns = torch.zeros_like(self.rewards[:self.size])
        
        # Compute GAE for each agent
        for agent_idx in range(self.num_agents):
            last_gae = 0
            
            for t in reversed(range(self.size)):
                # Get next value (either from buffer or last_values)
                if t == self.size - 1:
                    next_value = last_values[agent_idx]
                else:
                    next_value = self.values[t + 1, agent_idx]
                
                # Current value
                current_value = self.values[t, agent_idx]
                
                # Compute TD error
                delta = (
                    self.rewards[t, agent_idx] + 
                    gamma * next_value * (1 - self.dones[t, agent_idx]) - 
                    current_value
                )
                
                # Compute GAE
                last_gae = delta + gamma * gae_lambda * (1 - self.dones[t, agent_idx]) * last_gae
                self.advantages[t, agent_idx] = last_gae
                
                # Compute returns
                self.returns[t, agent_idx] = self.advantages[t, agent_idx] + current_value
        
        return self.advantages.detach(), self.returns.detach()

    def get_all(self) -> Tuple[torch.Tensor, ...]:
        """Get all transitions with shape verification"""
        data = (
            self.states[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.next_states[:self.size],
            self.dones[:self.size],
            self.logprobs[:self.size],
            self.values[:self.size]
        )
        
        # Verify shapes before returning
        assert data[0].shape == (self.size, self.num_agents, self.state_dim)
        assert data[1].shape == (self.size, self.num_agents, self.action_dim)
        assert data[2].shape == (self.size, self.num_agents)
        assert data[3].shape == (self.size, self.num_agents, self.state_dim)
        assert data[4].shape == (self.size, self.num_agents)
        assert data[5].shape == (self.size, self.num_agents)
        assert data[6].shape == (self.size, self.num_agents)
        
        return data

    def get_agent_data(self, agent_idx: int) -> AgentBatch:
        batch = AgentBatch(
            states=self.states[:self.size, agent_idx],
            next_states=self.next_states[:self.size, agent_idx],
            actions=self.actions[:self.size, agent_idx],
            rewards=self.rewards[:self.size, agent_idx],
            dones=self.dones[:self.size, agent_idx],
            values=self.values[:self.size, agent_idx],
            logprobs=self.logprobs[:self.size, agent_idx],
            advantages=self.advantages[:self.size, agent_idx] if self.advantages is not None else None,
            returns=self.returns[:self.size, agent_idx] if self.returns is not None else None
        )
        
        return batch

    def clear(self):
        """Reset buffer and all tensors"""
        self.ptr = 0
        self.size = 0
        
        # Reset all tensors to zeros
        self.states = torch.zeros((self.buffer_size, self.num_agents, self.state_dim)).to(self.device)
        self.actions = torch.zeros((self.buffer_size, self.num_agents, self.action_dim)).to(self.device)
        self.rewards = torch.zeros((self.buffer_size, self.num_agents)).to(self.device)
        self.next_states = torch.zeros((self.buffer_size, self.num_agents, self.state_dim)).to(self.device)
        self.dones = torch.zeros((self.buffer_size, self.num_agents)).to(self.device)
        
        # Reset PPO-specific buffers
        self.logprobs = torch.zeros((self.buffer_size, self.num_agents)).to(self.device)
        self.values = torch.zeros((self.buffer_size, self.num_agents)).to(self.device)
        self.returns = torch.zeros((self.buffer_size, self.num_agents)).to(self.device)
        self.advantages = torch.zeros((self.buffer_size, self.num_agents)).to(self.device)
        
        # Verify shapes after clearing
        self._verify_shapes()

    def is_ready(self) -> bool:
        """Check if buffer has enough data for an update"""
        return self.size >= self.buffer_size