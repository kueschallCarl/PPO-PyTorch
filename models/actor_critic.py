import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    """Decentralized actor network that only sees its own observations"""
    def __init__(self, state_dim, action_dim, hidden_dim=64, has_continuous_action_space=True):
        super(Actor, self).__init__()
        
        self.has_continuous_action_space = has_continuous_action_space
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        self.final_activation = nn.Tanh() if has_continuous_action_space else nn.Softmax(dim=-1)
            
    def forward(self, state):
        return self.final_activation(self.network(state))

class CentralizedCritic(nn.Module):
    """Centralized critic network that sees all agents' observations and actions"""
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=64):
        super(CentralizedCritic, self).__init__()
        
        # Calculate input dimensions
        self.state_input_dim = state_dim * num_agents
        self.action_input_dim = action_dim * num_agents
        self.full_input_dim = self.state_input_dim + self.action_input_dim
        
        self.network = nn.Sequential(
            nn.Linear(self.full_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states, actions=None):
        """
        Args:
            states: [batch_size, num_agents, state_dim] or [num_agents, state_dim]
            actions: [batch_size, num_agents, action_dim] or [num_agents, action_dim]
        """
        if states.dim() == 2:
            states = states.unsqueeze(0)
        batch_size = states.size(0)
        
        states_flat = states.reshape(batch_size, -1)
        
        if actions is not None:
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)
            actions_flat = actions.reshape(batch_size, -1)
            inputs = torch.cat([states_flat, actions_flat], dim=1)
        else:
            action_padding = torch.zeros(batch_size, self.action_input_dim).to(states.device)
            inputs = torch.cat([states_flat, action_padding], dim=1)
        
        return self.network(inputs)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        
        self.has_continuous_action_space = has_continuous_action_space
        self.num_agents = num_agents
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # Decentralized actor (only sees its own observations)
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            has_continuous_action_space=has_continuous_action_space
        )
        
        # Centralized critic (sees all agents' observations and actions)
        self.critic = CentralizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents
        )
    
    def act(self, state, global_state, actions=None):
        """
        Decentralized execution: actor only uses its own observation
        Centralized critic: uses global state and all actions
        """
        action_mean = self.actor(state)
        
        if self.has_continuous_action_space:
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            dist = Categorical(action_mean)
            
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # Critic evaluation uses global information
        if actions is not None:
            state_val = self.critic(global_state, actions)
        else:
            state_val = self.critic(global_state)
            
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action, global_state, all_actions):
        """Evaluate the actor-critic networks
        Args:
            state: Current agent's state [batch, state_dim]
            action: Current agent's action [batch, action_dim]
            global_state: States of all agents [batch, num_agents, state_dim]
            all_actions: Actions of all agents [batch, num_agents, action_dim]
        Returns:
            action_logprobs: Log probabilities of actions
            state_value: Value estimate from critic
            dist_entropy: Entropy of action distribution
        """
        # Ensure inputs are tensors and have proper shape
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(device)
        if not isinstance(global_state, torch.Tensor):
            global_state = torch.FloatTensor(global_state).to(device)
        if not isinstance(all_actions, torch.Tensor):
            all_actions = torch.FloatTensor(all_actions).to(device)

        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if global_state.dim() == 2:
            global_state = global_state.unsqueeze(0)
        if all_actions.dim() == 2:
            all_actions = all_actions.unsqueeze(0)

        # Get value estimate
        state_value = self.critic(global_state, all_actions)

        # Get action distribution
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_value, dist_entropy