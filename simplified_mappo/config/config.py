from dataclasses import dataclass, field
from typing import Optional, List
import torch
import os
from datetime import datetime

@dataclass
class EnvConfig:
    """Configuration for the environment settings"""
    """Possible environments: simple_spread, simple_reference, simple_speaker_listener"""
    env_name: str = "simple"  # Name of the environment to train in
    num_agents: int = 1              # Number of agents in environment
    episode_length: int = 64         # Length of each episode
    max_episodes: int = 500        # Maximum number of episodes
    max_training_timesteps: int = int(1e4)  # Total training steps
    has_continuous_action_space: bool = True
    continuous_actions: bool = True
    num_landmarks: int = 1

@dataclass
class LogConfig:
    """Configuration for logging and saving models"""
    print_freq: Optional[int] = 5
    log_freq: Optional[int] = 1000
    save_model_freq: int = int(1e4)
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
    eval_frequency: int = 250

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.90
    gae_lambda: float = 0.93
    clip_ratio: float = 0.2
    entropy_coef: float = 0.02
    value_loss_coef: float = 0.5
    policy_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_gae: bool = True
    use_value_clipping: bool = True
    normalize_advantages: bool = True
    num_updates: int = 80
    action_std_decay_freq: int = 1000
    action_std_decay_rate: float = 0.005
    min_action_std: float = 0.1
    visualize_eval: bool = True  # Whether to visualize one evaluation episode
    eval_delay: float = 0.05      # Delay between steps during evaluation visualization
    action_std_init: float = 0.5
    use_action_scaling: bool = True
    action_scale: float = 2.0
    exploration_noise: float = 0.1

@dataclass
class Config:
    """Main configuration class"""
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LogConfig = field(default_factory=LogConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    algorithm: str = "ippo"  # or "ippo"
    seed: Optional[int] = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        pass

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
        
        # Generate run_name after all arguments have been processed
        if config.log.run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            seed_str = f"seed{config.seed}" if config.seed is not None else "noseed"
            config.log.run_name = (
                f"{config.algorithm}"
                f"_{config.env.env_name}"
                f"_{config.env.num_agents}agents"
                f"_{seed_str}"
                f"_{timestamp}"
            )
        
        return config