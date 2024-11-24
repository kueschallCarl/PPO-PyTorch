import os
from datetime import datetime
import torch
import numpy as np
from models.ppo import PPO
from config.config import Config
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import SharedReplayBuffer
from utils.env_factory import make_env

class MAPPO:
    def __init__(self, cfg: Config):
        # Initialize environment
        self.env, self.state_dim, self.action_dim = make_env(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        
        # Set up logging
        self.setup_logging()
        print(f"Writer initialized with log dir: {self.writer.log_dir}")
        
        # Initialize buffer and agents
        self.buffer = SharedReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.env.num_agents,
            buffer_size=int(cfg.env.max_ep_len * cfg.ppo.update_timestep),
            device=self.device
        )
        
        # Initialize agents with debug print
        self.agents = []
        for i in range(self.env.num_agents):
            agent = PPO(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                num_agents=self.env.num_agents,
                cfg=cfg,
                writer=self.writer
            )
            print(f"Agent {i} initialized with writer: {agent.writer is not None}")
            self.agents.append(agent)
        
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_updates = 0

    def setup_logging(self):
        """Set up tensorboard and checkpoint directories"""
        run_name = f"PPO_{self.cfg.env.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.checkpoint_dir = f"checkpoints/{run_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def select_actions(self, observations):
        """Select actions for all agents"""
        actions = {}
        logprobs = {}  # New dictionary for logprobs
        
        # Create global state
        global_states = torch.stack([
            torch.FloatTensor(obs).to(self.device) 
            for obs in observations.values()
        ])
        
        # Initialize previous actions
        prev_actions = torch.zeros(self.env.num_agents, self.action_dim).to(self.device)
        
        # Select actions for each agent
        for agent_id, obs in observations.items():
            agent_idx = int(agent_id.split('_')[1])
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            action, logprob = self.agents[agent_idx].select_action(
                state=obs_tensor,
                actions=prev_actions,
                global_state=global_states
            )
            
            actions[agent_id] = action
            logprobs[agent_id] = logprob  # Store logprob
            prev_actions[agent_idx] = torch.FloatTensor(action).to(self.device)
            
        return actions, logprobs

    def train(self):
        """Main training loop"""
        print("Starting training...")
        time_step = 0
        episode = 0
        
        # Log hyperparameters
        self.writer.add_text('hyperparameters', str(self.cfg), 0)
        
        while time_step < self.cfg.env.max_training_timesteps:
            episode_reward = 0
            episode_length = 0
            step_rewards = []
            
            # Add random seed for each episode and print it
            seed = episode + self.cfg.env.seed if hasattr(self.cfg.env, 'seed') else None
            print(f"Episode {episode} starting with seed: {seed}")
            observations = self.env.reset(seed=seed)[0]
            
            while True:
                # Select actions and get logprobs
                actions, logprobs = self.select_actions(observations)
                
                # Environment step
                next_obs, rewards, terms, truncs, _ = self.env.step(actions)
                
                # Store transition with logprobs
                self.buffer.add(
                    states=observations,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_obs,
                    dones={k: terms[k] or truncs[k] for k in terms},
                    logprobs=logprobs  # Add logprobs to buffer
                )
                
                observations = next_obs
                step_rewards.append(sum(rewards.values()))
                episode_reward += step_rewards[-1]
                episode_length += 1
                time_step += 1
                
                # Log step-level metrics
                self.writer.add_scalar('Training/step_reward', step_rewards[-1], time_step)
                self.writer.add_scalar('Training/buffer_size', self.buffer.size, time_step)
                
                if self.buffer.is_ready():
                    self.update()
                    self.total_updates += 1
                
                if all(terms.values()) or all(truncs.values()):
                    break
            
            # Episode completed - log episode-level metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Log episode statistics
            self.log_episode(episode, time_step, episode_reward, episode_length)
            
            # Log rolling statistics
            window_size = 100
            if len(self.episode_rewards) >= window_size:
                self.writer.add_scalar('Training/reward_mean_100', 
                                     np.mean(self.episode_rewards[-window_size:]), episode)
                self.writer.add_scalar('Training/reward_std_100', 
                                     np.std(self.episode_rewards[-window_size:]), episode)
                self.writer.add_scalar('Training/length_mean_100', 
                                     np.mean(self.episode_lengths[-window_size:]), episode)
            
            episode += 1
            
            # Save checkpoint if it's time
            if time_step % self.cfg.log.save_model_freq == 0:
                self.save_checkpoint()

    def update(self):
        """Update all agents"""
        # Get all data from buffer
        states, actions, rewards, next_states, dones, logprobs = self.buffer.get_all()
        
        # Compute last values for all agents
        with torch.no_grad():
            last_values = torch.zeros(self.env.num_agents).to(self.device)
            for agent_idx in range(self.env.num_agents):
                last_state = next_states[-1]
                last_values[agent_idx] = self.agents[agent_idx].policy.critic(
                    last_state.unsqueeze(0),
                    None
                ).squeeze()
        
        # Compute GAE and returns in buffer
        self.buffer.compute_gae_and_advantages(
            last_values=last_values,
            gamma=self.cfg.ppo.gamma,
            gae_lambda=self.cfg.ppo.gae_lambda
        )
        
        # Update each agent
        for agent_idx, agent in enumerate(self.agents):
            agent_batch = self.buffer.get_agent_data(agent_idx)
            
            _ = agent.update(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                agent_idx=agent_idx,
                agent_batch=agent_batch,
                logprobs=logprobs
            )
        
        self.buffer.clear()

    def log_episode(self, episode, time_step, reward, length):
        """Log episode statistics"""
        self.writer.add_scalar('Training/episode_reward', reward, episode)
        self.writer.add_scalar('Training/episode_length', length, episode)
        self.writer.add_scalar('Training/step_count', time_step, episode)
        
        # Additional metrics
        self.writer.add_scalar('Training/reward_per_step', reward/length, episode)
        
        # Progress metrics
        completion_pct = (time_step / self.cfg.env.max_training_timesteps) * 100
        self.writer.add_scalar('Training/completion_percentage', completion_pct, episode)
        
        print(f"Episode {episode} | Steps: {time_step} | Reward: {reward:.2f} | Length: {length} | Progress: {completion_pct:.1f}%")

    def save_checkpoint(self):
        """Save agent checkpoints"""
        for idx, agent in enumerate(self.agents):
            path = os.path.join(self.checkpoint_dir, f'agent_{idx}.pth')
            agent.save(path)

if __name__ == '__main__':
    cfg = Config()
    mappo = MAPPO(cfg)
    mappo.train()