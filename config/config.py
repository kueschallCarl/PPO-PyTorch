from dataclasses import dataclass
import torch

@dataclass
class ActionConfig:
    action_std: float = 0.5  # Initial standard deviation for continuous actions

@dataclass
class EnvConfig:
    env_name: str = "simple_v3"  # MPE environment name
    has_continuous_action_space: bool = True  # Whether action space is continuous
    continuous_actions: bool = True  # For PettingZoo environment initialization
    max_ep_len: int = 100  # Maximum episode length
    max_training_timesteps: int = int(1e5)  # Maximum number of training timesteps
    seed: int = 1  # Random seed

@dataclass
class PPOConfig:
    # PPO hyperparameters
    lr_actor: float = 3e-4  # Learning rate for actor
    lr_critic: float = 1e-3  # Learning rate for critic
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    eps_clip: float = 0.2  # PPO clip parameter
    K_epochs: int = 5  # Number of epochs to update policy
    
    # Loss coefficients
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy coefficient for exploration
    
    # Gradient clipping
    max_grad_norm: float = 0.25  # Changed from 0.5
    
    # Buffer and batch settings
    update_timestep: int = 1024  # Update policy every n timesteps
    buffer_size: int = 1024  # Size of the replay buffer (should match update_timestep)
    normalize_advantages: bool = True  # Whether to normalize advantages

@dataclass
class NetworkConfig:
    hidden_dim: int = 64  # Hidden dimension for neural networks
    activation: str = "tanh"  # Activation function

@dataclass
class LogConfig:
    # Logging and saving frequencies
    log_freq: int = 1000  # Log metrics every n timesteps
    save_model_freq: int = 100000  # Save model every n timesteps
    tensorboard_dir: str = "runs"  # Directory for tensorboard logs
    checkpoint_dir: str = "checkpoints"  # Directory for model checkpoints

@dataclass
class TestConfig:
    checkpoint_path: str = "checkpoints/latest"  # Path to checkpoint directory
    total_test_episodes: int = 10  # Number of test episodes
    render: bool = True  # Whether to render environment
    frame_delay: float = 0.1  # Delay between frames when rendering
    random_seed: int = 1  # Random seed for testing

@dataclass
class Config:
    # Main configuration class that combines all sub-configs
    env: EnvConfig = EnvConfig()
    ppo: PPOConfig = PPOConfig()
    action: ActionConfig = ActionConfig()
    network: NetworkConfig = NetworkConfig()
    log: LogConfig = LogConfig()
    
    # Device configuration
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __post_init__(self):
        # Ensure consistency between related parameters
        self.env.continuous_actions = self.env.has_continuous_action_space
