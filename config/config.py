from dataclasses import dataclass
from typing import Optional
import torch
import os
@dataclass
class EnvConfig:
    """Configuration for the environment settings"""
    env_name: str = "simple_v3"  # Name of the environment to train in
    max_ep_len: int = 1000       # Maximum steps per episode. Higher = longer episodes, more exploration
    max_training_timesteps: int = int(2e5)  # Total training steps. Higher = more training time, better convergence
    has_continuous_action_space: bool = True  # Whether actions are continuous (True) or discrete (False)
    continuous_actions: bool = True  # Specific flag for PettingZoo environments

@dataclass
class LogConfig:
    """Configuration for logging and saving models"""
    # Higher frequencies = more detailed tracking but slower training
    print_freq: Optional[int] = None      # How often to print training info
    log_freq: Optional[int] = None        # How often to log metrics
    save_model_freq: int = int(1e5)       # How often to save model checkpoints. Higher = fewer saves
    log_dir: str = "logs/PPO_logs"        # Directory for storing logs
    model_dir: str = "logs/PPO_preTrained"  # Directory for saving models
    tensorboard_dir: str = "runs"         # Directory for tensorboard logs
    run_name: str = "gae_implementation"   # Identifier for this training run

@dataclass
class ActionConfig:
    """Configuration for action space exploration"""
    action_std: float = 0.6                # Initial action noise. Higher = more exploration
    action_std_decay_rate: float = 0.05    # How quickly to reduce exploration. Higher = faster reduction
    min_action_std: float = 0.1            # Minimum exploration noise. Higher = never fully exploits
    action_std_decay_freq: int = int(1e4)  # How often to decay exploration. Lower = faster adaptation

@dataclass
class PPOConfig:
    """Configuration for PPO algorithm parameters"""
    K_epochs: int = 80          # Policy update iterations. Higher = more stable but slower training
    eps_clip: float = 0.2       # PPO clipping parameter. Higher = larger policy updates
    gamma: float = 0.905         # Discount factor. Higher = more emphasis on future rewards
    gae_lambda: float = 0.93    # GAE parameter. Higher = more emphasis on long-term advantages
    use_gae: bool = True        # Whether to use Generalized Advantage Estimation
    use_value_clipping: bool = True  # Whether to use value function clipping
    lr_actor: float = 0.0003    # Actor learning rate. Higher = faster learning but potential instability
    lr_critic: float = 0.0003   # Critic learning rate. Higher = faster value estimation but potential instability
    update_timestep: float = 1  # Number of episodes before updating the policy (example: 4 episodes -> update every 4 * max_ep_len steps -> 4 * 1000 = 4000 steps)
    entropy_coef: float = 0.01  # Entropy coefficient. Higher = more exploration
    random_seed: int = 0        # Seed for reproducibility
    max_grad_norm = 0.5
    policy_loss_coef = 1.0
    value_loss_coef = 0.5
    normalize_advantages = True
    value_reg_coef: float = 0.01  # Value function regularization coefficient
    critic_clip_coef: float = 0.2  # Separate clip coefficient for critic gradients

@dataclass
class Config:
    env: EnvConfig = EnvConfig()
    log: LogConfig = LogConfig()
    action: ActionConfig = ActionConfig()
    ppo: PPOConfig = PPOConfig()
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Set dependent parameters
        if self.log.print_freq is None:
            self.log.print_freq = self.env.max_ep_len * 10
        if self.log.log_freq is None:
            self.log.log_freq = self.env.max_ep_len * 2 

@dataclass
class TestConfig:
    total_test_episodes: int = 100
    render: bool = True
    frame_delay: float = 0.3  # Delay between frames when rendering (0.0 for no delay)
    checkpoint_path: str = None  # Will be set in __post_init__
    random_seed: int = 0

    def __post_init__(self):
        if self.checkpoint_path is None:
            # Default path based on training configuration
            self.checkpoint_path = os.path.join(
                "logs/PPO_preTrained",
                "simple_v3",
                f"PPO_simple_v3_0_0_gae_implementation_20241119_013027.pth"
            )