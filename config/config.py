from dataclasses import dataclass
from typing import Optional
import torch
import os
@dataclass
class EnvConfig:
    """Configuration for the environment settings"""
    env_name: str = "simple_v3"  # Name of the environment to train in
    max_ep_len: int = 100       # Maximum steps per episode. Higher = longer episodes, more exploration
    max_training_timesteps: int = int(1e5)  # Total training steps. Higher = more training time, better convergence
    has_continuous_action_space: bool = True  # Whether actions are continuous (True) or discrete (False)
    continuous_actions: bool = True  # Specific flag for PettingZoo environments

@dataclass
class LogConfig:
    """Configuration for logging and saving models"""
    # Higher frequencies = more detailed tracking but slower training
    print_freq: Optional[int] = None      # How often to print training info
    log_freq: Optional[int] = None        # How often to log metrics
    save_model_freq: int = int(5e4)       # How often to save model checkpoints. Higher = fewer saves
    log_dir: str = "logs/PPO_logs"        # Directory for storing logs
    model_dir: str = "logs/PPO_preTrained"  # Directory for saving models
    tensorboard_dir: str = "runs"         # Directory for tensorboard logs
    run_name: str = "MAPPO_start"   # Identifier for this training run

@dataclass
class ActionConfig:
    """Configuration for action space exploration"""
    action_std: float = 0.7               # Initial action noise. Higher = more exploration
    action_std_decay_rate: float = 0.05    # How quickly to reduce exploration. Higher = faster reduction
    min_action_std: float = 0.1            # Minimum exploration noise. Higher = never fully exploits
    action_std_decay_freq: int = int(5e3)  # How often to decay exploration. Lower = faster adaptation

@dataclass
class PPOConfig:
    """Configuration for PPO algorithm parameters"""
    # Training Parameters
    K_epochs: int = 40         # Policy update iterations. Higher = more stable but slower training
    update_timestep: int = 4    # Changed from float to int
    random_seed: Optional[int] = None  # Changed from 0 to None to enable random initialization

    # Clipping and Regularization
    eps_clip: float = 0.15       # PPO clipping parameter. Higher = larger policy updates
    critic_clip_coef: float = 0.15  # Separate clip coefficient for critic gradients
    value_reg_coef: float = 0.001  # Value function regularization coefficient

    # Learning Rates
    lr_actor: float = 0.0001    # Actor learning rate. Higher = faster learning but potential instability
    lr_critic: float = 0.0001   # Critic learning rate. Higher = faster value estimation but potential instability

    # Hyperparameters
    gamma: float = 0.905         # Discount factor. Higher = more emphasis on future rewards
    gae_lambda: float = 0.93    # GAE parameter. Higher = more emphasis on long-term advantages
    entropy_coef: float = 0.01  # Entropy coefficient. Higher = more exploration

    # Model Parameters
    use_gae: bool = True        # Whether to use Generalized Advantage Estimation
    use_value_clipping: bool = True  # Whether to use value function clipping
    use_centralized_critic: bool = True
    critic_hidden_dim: int = 64
    critic_num_layers: int = 2

    # Buffer Size
    buffer_size: int = 2048      # Match update_timestep for simplicity

    # Loss Coefficients
    policy_loss_coef: float = 1.0
    value_loss_coef: float = 0.5
    normalize_advantages: bool = True
    max_grad_norm: float = 0.5

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
    frame_delay: float = 0.001  # Delay between frames when rendering (0.0 for no delay)
    checkpoint_path: str = None  # Will be set in __post_init__
    random_seed: int = 0

    def __post_init__(self):
        if self.checkpoint_path is None:
            # Default path based on training configuration
            self.checkpoint_path = "runs/PPO_simple_v3_None_0_MAPPO_start_20241124_020515"
