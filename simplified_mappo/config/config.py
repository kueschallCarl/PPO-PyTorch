from dataclasses import dataclass, field
from typing import Optional, List
import torch
import os
from datetime import datetime

@dataclass
class EnvConfig:
    """Configuration for the environment settings"""
    """Possible environments: simple_spread, simple_reference, simple_speaker_listener"""
    env_name: str = "simple_spread"  # Name of the environment to train in
    num_agents: int = 3              # Number of agents in environment
    episode_length: int = 25         # Length of each episode
    max_episodes: int = 1000         # Maximum number of episodes
    max_training_timesteps: int = int(1e4)  # Total training steps
    has_continuous_action_space: bool = True
    continuous_actions: bool = True

@dataclass
class LogConfig:
    """Configuration for logging and saving models"""
    print_freq: Optional[int] = 1000
    log_freq: Optional[int] = 1000
    save_model_freq: int = int(5e4)
    log_dir: str = "logs"
    model_dir: str = "models"
    wandb_project: str = "simplified-mappo-ippo"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    use_wandb: bool = True
    tensorboard_dir: str = "runs"
@dataclass
class BufferConfig:
    """Configuration for replay buffer"""
    size: int = 2048
    batch_size: int = 64
    advantage_normalization: bool = True

@dataclass
class PolicyConfig:
    """Configuration for policy networks"""
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"
    initialization: str = "orthogonal"
    gain: float = 0.01
    action_std: float = 0.5

@dataclass
class TrainingConfig:
    """Shared training parameters"""
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    policy_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_gae: bool = True
    use_value_clipping: bool = True
    normalize_advantages: bool = True
    num_updates: int = 10
    eval_frequency: int = 100
    action_std_decay_freq: int = 10000
    action_std_decay_rate: float = 0.01
    min_action_std: float = 0.1
@dataclass
class Config:
    """Main configuration class"""
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LogConfig = field(default_factory=LogConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    algorithm: str = "mappo"  # or "mappo"
    seed: Optional[int] = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.log.run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            seed_str = f"seed{self.seed}" if self.seed is not None else "noseed"
            self.log.run_name = (
                f"{self.algorithm}"
                f"_{self.env.env_name}"
                f"_{self.env.num_agents}agents"
                f"_{seed_str}"
                f"_{timestamp}"
            )

    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        config = cls()
        
        # Update config with any non-None values from args
        for key, value in vars(args).items():
            if value is not None:
                # Handle nested configs
                if '.' in key:
                    section, param = key.split('.')
                    setattr(getattr(config, section), param, value)
                else:
                    setattr(config, key, value)
        
        return config