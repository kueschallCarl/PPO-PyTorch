import torch
import torch.nn as nn
from models.actor_critic import ActorCritic, Critic, CentralizedCritic
from utils.buffer import SharedReplayBuffer
from config.config import Config
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PPO:
    def __init__(self, state_dim, action_dim, cfg: Config, writer: SummaryWriter = None, critic_type='centralized', num_agents=1):
        self.has_continuous_action_space = cfg.env.has_continuous_action_space
        self.action_std = cfg.action.action_std if self.has_continuous_action_space else None
        self.device = cfg.device
        
        # Add PPO hyperparameters as attributes
        self.gamma = cfg.ppo.gamma
        self.eps_clip = cfg.ppo.eps_clip
        self.K_epochs = cfg.ppo.K_epochs
        self.gae_lambda = cfg.ppo.gae_lambda
        self.entropy_coef = cfg.ppo.entropy_coef
        self.value_loss_coef = cfg.ppo.value_loss_coef
        self.max_grad_norm = cfg.ppo.max_grad_norm
        self.normalize_advantages = cfg.ppo.normalize_advantages
        
        # Store dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Initialize action variance
        if self.has_continuous_action_space:
            self.action_var = torch.full((action_dim,), self.action_std * self.action_std).to(self.device)
        
        # Initialize buffer
        self.buffer = SharedReplayBuffer(state_dim, action_dim, num_agents, cfg.ppo.buffer_size, cfg.device)
        
        # Initialize actor-critic
        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            has_continuous_action_space=self.has_continuous_action_space,
            action_std_init=self.action_std,
            critic_type=critic_type,
            num_agents=num_agents
        ).to(self.device)
        
        self.policy_old = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            has_continuous_action_space=self.has_continuous_action_space,
            action_std_init=self.action_std,
            critic_type=critic_type,
            num_agents=num_agents
        ).to(self.device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': cfg.ppo.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': cfg.ppo.lr_critic}
        ])

        self.MseLoss = nn.MSELoss()

        # Use provided writer or create new one
        self.writer = writer or SummaryWriter(os.path.join(cfg.log.tensorboard_dir, 
                                        f"{cfg.env.env_name}_{cfg.log.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        
        # Only add graph visualization if not using centralized critic
        if writer and critic_type != 'centralized':
            # Log network graph
            dummy_state = torch.zeros(1, state_dim).to(cfg.device)
            self.writer.add_graph(self.policy, dummy_state)
        elif writer:
            # For centralized critic, we'll skip the graph visualization
            print("Skipping network visualization for centralized critic")

    def set_action_std(self, new_action_std):
        """Set the action standard deviation"""
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("WARNING: Calling PPO::set_action_std() on discrete action space policy")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """Decay the action standard deviation"""
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING: Calling PPO::decay_action_std() on discrete action space policy")

    def select_action(self, state, actions=None, global_state=None, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                if self.has_continuous_action_space:
                    action_mean = self.policy_old.actor(state)
                    action = action_mean
                else:
                    action_probs = self.policy_old.actor(state)
                    action = torch.argmax(action_probs)
                
                # Get state value for logging
                if self.policy.critic_type == 'centralized':
                    if actions is None or global_state is None:
                        # During initial action selection, we might not have actions
                        # Use zeros as placeholder actions
                        batch_size = state.shape[0]
                        dummy_actions = torch.zeros(batch_size, self.policy.num_agents, 
                                                 self.action_dim).to(self.device)
                        state_val = self.policy_old.critic(global_state, dummy_actions)
                    else:
                        state_val = self.policy_old.critic(global_state, actions)
                else:
                    state_val = self.policy_old.critic(state)
            else:
                # Stochastic action selection (training mode)
                if self.policy.critic_type == 'centralized':
                    action, action_logprob, state_val = self.policy_old.act(
                        state, 
                        actions=actions, 
                        global_state=global_state
                    )
                else:
                    action, action_logprob, state_val = self.policy_old.act(state)

            # Add checks for NaN values
            if torch.isnan(action).any():
                print(f"Warning: NaN detected in action: {action}")
                action = torch.nan_to_num(action, 0.5)

            if self.has_continuous_action_space:
                action_np = action.detach().cpu().numpy().flatten()
                if np.isnan(action_np).any():
                    action_np = np.nan_to_num(action_np, 0.5)
                return np.clip(action_np, 0.0, 1.0)
            else:
                return action.item()

    def compute_gae(self, rewards, values, dones, next_value=0):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        next_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                next_advantage = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.cfg.ppo.gae_lambda * next_advantage * (1 - dones[t])
            
            next_advantage = advantages[t]
            next_value = values[t]
        
        return advantages

    def compute_simple_advantage(self, rewards, values, dones):
        """
        Compute simple advantage using discounted returns - value estimates
        """
        returns = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        advantages = returns - values
        
        return advantages, returns

    def update(self):
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Convert to numpy for advantage calculation
        rewards_np = np.array([r for r in self.buffer.rewards])
        values_np = old_state_values.cpu().numpy()
        dones_np = np.array([d for d in self.buffer.is_terminals])
        
        # Calculate advantages based on config
        if self.cfg.ppo.use_gae:
            advantages_np = self.compute_gae(rewards_np, values_np, dones_np)
            advantages = torch.FloatTensor(advantages_np).to(device)
            returns = advantages + old_state_values
        else:
            advantages, returns = self.compute_simple_advantage(
                self.buffer.rewards, 
                old_state_values,
                self.buffer.is_terminals
            )
        
        # Normalize advantages
        if self.cfg.ppo.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track statistics
        avg_loss = 0
        avg_value_loss = 0
        avg_policy_loss = 0
        avg_entropy = 0

        for epoch in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Calculate probability ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Value loss calculation with optional clipping
            if self.cfg.ppo.use_value_clipping:
                value_pred_clipped = old_state_values + torch.clamp(
                    state_values - old_state_values,
                    -self.eps_clip * old_state_values.abs(),  # Scale clipping with value magnitude
                    self.eps_clip * old_state_values.abs()
                )
                value_losses = (state_values - returns).pow(2)
                value_losses_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (state_values - returns).pow(2).mean()

            # Add value function regularization
            value_reg_loss = 0.01 * state_values.pow(2).mean()  # L2 regularization
            value_loss += value_reg_loss

            # Calculate losses with coefficients
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coef * dist_entropy.mean()
            
            # Combine losses with proper coefficients
            loss = (
                policy_loss * self.cfg.ppo.policy_loss_coef + 
                value_loss * self.cfg.ppo.value_loss_coef + 
                entropy_loss
            )
            
            # Gradient update with clipping
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients separately for actor and critic
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.cfg.ppo.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.cfg.ppo.max_grad_norm * 0.5)  # Lower clip for critic
            
            self.optimizer.step()

            # Accumulate statistics
            avg_loss += loss.item()
            avg_value_loss += value_loss.item()
            avg_policy_loss += policy_loss.item()
            avg_entropy += dist_entropy.mean().item()

        # Log statistics to tensorboard
        steps = len(self.buffer.rewards)
        self.total_steps = getattr(self, 'total_steps', 0) + steps
        
        # Log average losses
        self.writer.add_scalar('Loss/total', avg_loss / self.K_epochs, self.total_steps)
        self.writer.add_scalar('Loss/value', avg_value_loss / self.K_epochs, self.total_steps)
        self.writer.add_scalar('Loss/policy', avg_policy_loss / self.K_epochs, self.total_steps)
        self.writer.add_scalar('Policy/entropy', avg_entropy / self.K_epochs, self.total_steps)
        
        # Log policy statistics
        self.writer.add_scalar('Policy/mean_ratio', ratios.mean().item(), self.total_steps)
        self.writer.add_scalar('Policy/mean_advantage', advantages.mean().item(), self.total_steps)
        
        # Log value statistics
        self.writer.add_scalar('Value/mean_value', state_values.mean().item(), self.total_steps)
        self.writer.add_scalar('Value/value_std', state_values.std().item(), self.total_steps)
        
        # Log histograms
        self.writer.add_histogram('Policy/action_logprobs', logprobs.detach(), self.total_steps)
        self.writer.add_histogram('Policy/advantages', advantages.detach(), self.total_steps)
        self.writer.add_histogram('Value/values', state_values.detach(), self.total_steps)
        
        # Log network parameters
        for name, param in self.policy.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param.data, self.total_steps)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, self.total_steps)

        # Log value function specific metrics
        self.writer.add_scalar('Value/mean_value_change', (state_values - old_state_values).abs().mean().item(), self.total_steps)
        if self.cfg.ppo.use_value_clipping:
            self.writer.add_scalar('Value/clipped_fraction', 
                (value_losses_clipped < value_losses).float().mean().item(), 
                self.total_steps)
            self.writer.add_scalar('Value/clipping_threshold', 
                self.eps_clip, 
                self.total_steps)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True)
        )

    def update_mappo(self, states, actions, rewards, next_states, dones, agent_idx):
        """Update policy for MAPPO"""
        # Reshape tensors to have batch dimension first
        batch_size = states.size(0)
        
        # Get data for current agent
        agent_states = states[:, agent_idx]  # [batch_size, state_dim]
        agent_actions = actions[:, agent_idx]  # [batch_size, action_dim]
        agent_rewards = rewards[:, agent_idx]  # [batch_size]
        agent_next_states = next_states[:, agent_idx]  # [batch_size, state_dim]
        agent_dones = dones[:, agent_idx]  # [batch_size]
        
        # Get old action log probabilities
        with torch.no_grad():
            if self.has_continuous_action_space:
                action_mean = self.policy_old.actor(agent_states)
                action_var = self.action_var.expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, action_var.sqrt())
                old_action_logprobs = dist.log_prob(agent_actions).sum(dim=-1)  # Sum across action dimensions
            else:
                action_probs = self.policy_old.actor(agent_states)
                dist = torch.distributions.Categorical(action_probs)
                old_action_logprobs = dist.log_prob(agent_actions)
        
        # Compute returns and advantages
        with torch.no_grad():
            # Get values for current states using centralized critic
            values = self.policy.critic(states, actions).squeeze(-1)  # [batch_size]
            
            # Get values for next states
            next_actions = torch.zeros_like(actions)  # Placeholder for next actions
            next_values = self.policy.critic(next_states, next_actions).squeeze(-1)  # [batch_size]
            
            # Compute returns and advantages
            returns = torch.zeros(batch_size, device=self.device)
            advantages = torch.zeros(batch_size, device=self.device)
            gae = 0
            
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    next_value = next_values[t]
                else:
                    next_value = values[t + 1]
                    
                delta = agent_rewards[t] + self.gamma * next_value * (1 - agent_dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - agent_dones[t]) * gae
                
                returns[t] = gae + values[t]
                advantages[t] = gae
                
            # Normalize advantages
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track statistics
        avg_loss = 0
        avg_value_loss = 0
        avg_policy_loss = 0
        avg_entropy = 0
        avg_clip_fraction = 0
        avg_approx_kl = 0
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Get current policy distributions
            if self.has_continuous_action_space:
                action_mean = self.policy.actor(agent_states)
                action_var = self.action_var.expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, action_var.sqrt())
                action_logprobs = dist.log_prob(agent_actions).sum(dim=-1)  # Sum across action dimensions
            else:
                action_probs = self.policy.actor(agent_states)
                dist = torch.distributions.Categorical(action_probs)
                action_logprobs = dist.log_prob(agent_actions)
                
            # Get entropy
            dist_entropy = dist.entropy().mean()
            
            # Get state values from critic
            state_values = self.policy.critic(states, actions).squeeze(-1)
            
            # Calculate ratios and surrogate losses
            ratios = torch.exp(action_logprobs - old_action_logprobs)  # [batch_size]
            surr1 = ratios * advantages  # [batch_size]
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages  # [batch_size]
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((returns - state_values) ** 2).mean()
            entropy_loss = -self.entropy_coef * dist_entropy.mean()
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
            
            # Update policy
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Calculate clip fraction
            clip_fraction = ((ratios - 1.0).abs() > self.eps_clip).float().mean()
            
            # Calculate approximate KL divergence
            approx_kl = (old_action_logprobs - action_logprobs).mean()
            
            # Accumulate statistics
            avg_loss += total_loss.item()
            avg_value_loss += value_loss.item()
            avg_policy_loss += policy_loss.item()
            avg_entropy += dist_entropy.mean().item()
            avg_clip_fraction += clip_fraction.item()
            avg_approx_kl += approx_kl.item()
        
        # Return average losses and metrics
        return {
            'total_loss': avg_loss / self.K_epochs,
            'value_loss': avg_value_loss / self.K_epochs,
            'policy_loss': avg_policy_loss / self.K_epochs,
            'entropy_loss': avg_entropy / self.K_epochs,
            'clip_fraction': avg_clip_fraction / self.K_epochs,
            'approx_kl': avg_approx_kl / self.K_epochs,
            'mean_value': state_values.mean().item(),
            'value_std': state_values.std().item(),
            'mean_ratio': ratios.mean().item(),
            'mean_advantage': advantages.mean().item()
        }