import torch
import torch.nn as nn
from models.actor_critic import ActorCritic
from utils.buffer import RolloutBuffer
from config.config import Config
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PPO:
    def __init__(self, state_dim, action_dim, cfg: Config):
        self.cfg = cfg
        self.has_continuous_action_space = cfg.env.has_continuous_action_space
        
        if self.has_continuous_action_space:
            self.action_std = cfg.action.action_std
        
        self.gamma = cfg.ppo.gamma
        self.eps_clip = cfg.ppo.eps_clip
        self.K_epochs = cfg.ppo.K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            has_continuous_action_space=self.has_continuous_action_space,
            action_std_init=cfg.action.action_std
        ).to(cfg.device)
        
        self.policy_old = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            has_continuous_action_space=self.has_continuous_action_space,
            action_std_init=cfg.action.action_std
        ).to(cfg.device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': cfg.ppo.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': cfg.ppo.lr_critic}
        ])

        self.MseLoss = nn.MSELoss()

        # Add tensorboard writer
        self.writer = SummaryWriter(os.path.join(cfg.log.tensorboard_dir, 
                                                f"{cfg.env.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        
        # Log network graph
        dummy_state = torch.zeros(1, state_dim).to(cfg.device)
        self.writer.add_graph(self.policy, dummy_state)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()

        # Track statistics
        avg_loss = 0
        avg_value_loss = 0
        avg_policy_loss = 0
        avg_entropy = 0

        for epoch in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Split losses for logging
            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy
            
            loss = policy_loss + value_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # Accumulate statistics
            avg_loss += loss.mean().item()
            avg_value_loss += value_loss.mean().item()
            avg_policy_loss += policy_loss.mean().item()
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

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))