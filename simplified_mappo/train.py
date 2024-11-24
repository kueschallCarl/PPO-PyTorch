import torch
from algorithms.actor_critic import MLPPolicy
from algorithms.mappo import MAPPO
from utils.buffer import SharedReplayBuffer
from runner import Runner
import logging
import time
from datetime import datetime
import traceback
import wandb
from dataclasses import asdict, dataclass, field
from typing import Optional, List
import os

@dataclass
class EnvConfig:
    """Configuration for the environment settings"""
    env_name: str = "simple_spread"  # Name of the environment to train in
    num_agents: int = 3              # Number of agents in environment
    episode_length: int = 25         # Length of each episode
    max_episodes: int = 1000         # Maximum number of episodes
    max_training_timesteps: int = int(1e5)  # Total training steps
    has_continuous_action_space: bool = True
    continuous_actions: bool = True

@dataclass
class LogConfig:
    """Configuration for logging and saving models"""
    print_freq: Optional[int] = None
    log_freq: Optional[int] = None
    save_model_freq: int = int(5e4)
    log_dir: str = "logs"
    model_dir: str = "models"
    tensorboard_dir: str = "runs"
    wandb_project: str = "simplified-mappo-implementation"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    use_wandb: bool = True

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
    max_grad_norm: float = 0.5
    use_gae: bool = True
    normalize_advantages: bool = True
    num_updates: int = 10
    eval_frequency: int = 100

@dataclass
class Config:
    """Main configuration class"""
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LogConfig = field(default_factory=LogConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    algorithm: str = "ippo"  # or "mappo"
    seed: Optional[int] = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.log.run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log.run_name = f"{self.algorithm}_{self.env.env_name}_{timestamp}"

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

def train_mappo(cfg: Config):
    # Initialize wandb if enabled
    if cfg.log.use_wandb:
        wandb.init(
            project=cfg.log.wandb_project,
            entity=cfg.log.wandb_entity,
            name=cfg.log.run_name,
            config=asdict(cfg)
        )
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        logging.info(f"Starting training with config:")
        logging.info(f"Scenario: {cfg.env.env_name}, Agents: {cfg.env.num_agents}")
        logging.info(f"Episodes: {cfg.env.max_episodes}, Episode length: {cfg.env.episode_length}")
        logging.info(f"Buffer size: {cfg.buffer.size}, Batch size: {cfg.buffer.batch_size}, LR: {cfg.training.lr_actor}")

        # Initialize device
        device = torch.device(cfg.device)
        logging.info(f"Using device: {device}")
        
        # Initialize environment and runner
        runner = Runner(cfg.env.env_name, cfg.env.num_agents, device=device)
        
        # Initialize policy and algorithm
        policy = MLPPolicy(
            obs_space=runner.obs_space,
            action_space=runner.action_space
        ).to(device)
        
        algorithm = MAPPO(
            policy=policy,
            lr=cfg.training.lr_actor
        )
        
        # Initialize buffer
        buffer = SharedReplayBuffer(
            num_agents=cfg.env.num_agents,
            obs_space=runner.obs_space,
            act_space=runner.action_space,
            size=cfg.buffer.size,
            device=device
        )
        
        # Training loop
        start_time = time.time()
        for episode in range(cfg.env.max_episodes):
            episode_start = time.time()
            
            try:
                # Collect experience
                episode_reward = runner.collect_episodes(policy, buffer, cfg.env.episode_length)
                
                # Track training metrics
                episode_metrics = {
                    "train/episode_reward": episode_reward,
                    "train/episode_duration": time.time() - episode_start,
                }
                
                # Update policy
                policy_metrics = {}
                for sample in buffer.get_samples(cfg.buffer.batch_size):
                    update_info = algorithm.update(sample)
                    # Aggregate policy update metrics
                    for k, v in update_info.items():
                        if k not in policy_metrics:
                            policy_metrics[k] = []
                        policy_metrics[k].append(v)
                
                # Average policy metrics over all updates
                for k, v in policy_metrics.items():
                    episode_metrics[f"train/{k}"] = sum(v) / len(v)
                
                # Log to wandb
                wandb.log(episode_metrics, step=episode)
                
                # Log progress
                episode_duration = time.time() - episode_start
                if (episode + 1) % 10 == 0:
                    logging.info(f"Episode {episode + 1}/{cfg.env.max_episodes} completed in {episode_duration:.2f}s. "
                               f"Reward: {episode_reward:.2f}")
                
                # Evaluate policy
                if (episode + 1) % cfg.training.eval_frequency == 0:
                    logging.info("Starting evaluation...")
                    try:
                        eval_reward = runner.eval_policy(policy)
                        eval_metrics = {
                            "eval/reward": eval_reward,
                            "eval/reward_diff": eval_reward - episode_reward
                        }
                        wandb.log(eval_metrics, step=episode)
                        
                        logging.info(f"Evaluation at episode {episode + 1}: "
                                   f"Training reward: {episode_reward:.2f}, "
                                   f"Eval reward: {eval_reward:.2f}")
                    except Exception as e:
                        logging.error(f"Error during evaluation: {str(e)}")
                        logging.error(traceback.format_exc())
            
            except Exception as e:
                logging.error(f"Error during episode {episode + 1}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        total_time = time.time() - start_time
        wandb.log({"train/total_time": total_time})
        logging.info(f"Training completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Fatal error during training: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    finally:
        wandb.finish() 