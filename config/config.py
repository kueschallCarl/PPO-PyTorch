from dataclasses import dataclass
from typing import Optional
import torch
import os
@dataclass
class EnvConfig:
    env_name: str = "simple_v3"
    max_ep_len: int = 1000
    max_training_timesteps: int = int(1e5) #= 100000
    has_continuous_action_space: bool = True
    continuous_actions: bool = True  # for PettingZoo env

@dataclass
class LogConfig:
    print_freq: Optional[int] = None  # will be set based on max_ep_len
    log_freq: Optional[int] = None    # will be set based on max_ep_len
    save_model_freq: int = int(1e5)
    log_dir: str = "logs/PPO_logs"
    model_dir: str = "logs/PPO_preTrained"
    tensorboard_dir: str = "runs"
    run_name: str = "gae_implementation"

@dataclass
class ActionConfig:
    action_std: float = 0.6
    action_std_decay_rate: float = 0.05
    min_action_std: float = 0.1
    action_std_decay_freq: int = int(2.5e5)

@dataclass
class PPOConfig:
    K_epochs: int = 80
    eps_clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_gae: bool = False
    lr_actor: float = 0.0003
    lr_critic: float = 0.001
    random_seed: int = 0

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
    total_test_episodes: int = 10
    render: bool = True
    frame_delay: float = 0.0  # Delay between frames when rendering (0.0 for no delay)
    checkpoint_path: str = None  # Will be set in __post_init__
    random_seed: int = 0

    def __post_init__(self):
        if self.checkpoint_path is None:
            # Default path based on training configuration
            self.checkpoint_path = os.path.join(
                "logs/PPO_preTrained",
                "simple_v3",
                f"PPO_simple_v3_0_0_gae_implementation_20241116_182524.pth"
            )