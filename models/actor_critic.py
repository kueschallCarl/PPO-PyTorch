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
        
        # Add final activation based on action space
        if has_continuous_action_space:
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Softmax(dim=-1)
            
    def forward(self, state):
        action_output = self.network(state)
        return self.final_activation(action_output)

class Critic(nn.Module):
    """Standard critic network for IPPO"""
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class CentralizedCritic(nn.Module):
    """Centralized critic network that sees all agents' observations and actions"""
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=64):
        super(CentralizedCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Calculate input dimension
        self.input_dim = (state_dim + action_dim) * num_agents
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states, actions):
        """
        Args:
            states: Tensor of shape [batch_size, num_agents, state_dim] or [num_agents, state_dim]
            actions: Tensor of shape [batch_size, num_agents, action_dim] or [num_agents, action_dim]
        """
        # Add batch dimension if not present
        if states.dim() == 2:
            states = states.unsqueeze(0)  # [1, num_agents, state_dim]
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)  # [1, num_agents, action_dim]
            
        batch_size = states.size(0)
        
        # Reshape inputs
        states_flat = states.reshape(batch_size, self.num_agents * self.state_dim)
        actions_flat = actions.reshape(batch_size, self.num_agents * self.action_dim)
        
        # Concatenate along feature dimension
        inputs = torch.cat([states_flat, actions_flat], dim=1)
        
        return self.network(inputs)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, 
                 critic_type='centralized', num_agents=1):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.critic_type = critic_type
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # Actor network (decentralized)
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            has_continuous_action_space=has_continuous_action_space
        )

        # Critic network (can be centralized or decentralized)
        if critic_type == 'centralized':
            self.critic = CentralizedCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_agents,
                hidden_dim=64
            )
        else:
            self.critic = Critic(state_dim=state_dim, hidden_dim=64)
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")

    def forward(self, state, actions=None, global_state=None):
        """
        Forward pass handling both centralized and decentralized critics
        """
        # Actor forward pass (always decentralized)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            if self.critic_type == 'centralized' and actions is not None and global_state is not None:
                state_value = self.critic(global_state, actions)
            else:
                state_value = self.critic(state)
            return action_mean, state_value
        else:
            action_probs = self.actor(state)
            if self.critic_type == 'centralized' and actions is not None and global_state is not None:
                state_value = self.critic(global_state, actions)
            else:
                state_value = self.critic(state)
            return action_probs, state_value
    
    def act(self, state, actions=None, global_state=None):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # Handle centralized critic
        if self.critic_type == 'centralized' and actions is not None and global_state is not None:
            state_val = self.critic(global_state, actions)
        else:
            state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action, actions=None, global_state=None):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        # Handle centralized critic
        if self.critic_type == 'centralized' and actions is not None and global_state is not None:
            state_values = self.critic(global_state, actions)
        else:
            state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy