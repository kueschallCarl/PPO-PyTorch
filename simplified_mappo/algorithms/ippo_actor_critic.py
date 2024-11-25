import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim

        if has_continuous_action_space:
            self.log_std = nn.Parameter(torch.log(torch.full((action_dim,), action_std_init)))
            self.action_std = action_std_init  # Keep track of current std

        # actor with ReLU and no final Tanh
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        # critic remains the same
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            std = self.log_std.exp()
            
            # Use independent Normal distributions with proper scaling
            dist = torch.distributions.Normal(action_mean, std)
            
            # During training, sample from distribution
            action = dist.rsample()  # Use rsample for reparameterization
            
            # Compute log probability with proper scaling
            action_logprob = dist.log_prob(action).sum(-1)
            
            # Ensure actions are properly scaled
            action = torch.tanh(action)  # Squash to [-1, 1]
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            std = self.log_std.exp()
            
            # Use independent Normal distributions
            dist = torch.distributions.Normal(action_mean, std)
            action_logprobs = dist.log_prob(action).sum(-1)
            dist_entropy = dist.entropy().sum(-1)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

    def set_action_std(self, new_action_std):
        """
        Update the action standard deviation
        """
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.log_std.data = torch.log(torch.full((self.action_dim,), new_action_std)).to(self.log_std.device)
        else:
            print("WARNING: Calling ActorCritic::set_action_std() on discrete action space policy")

    def get_action_std(self):
        """
        Get the current action standard deviation
        """
        if self.has_continuous_action_space:
            return self.log_std.exp().detach()
        else:
            print("WARNING: Calling ActorCritic::get_action_std() on discrete action space policy")
            return None