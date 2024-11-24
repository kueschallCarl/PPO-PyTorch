from dataclasses import dataclass
from typing import Optional

@dataclass
class TestConfig:
    """Configuration for testing trained models"""
    model_path: str = "simplified_mappo/runs/PPO_simple_spread_None_mappo_simple_spread_3agents_noseed_20241124_210343_20241124_210344/model.pth"                  # Path to the model checkpoint
    config_path: str = "simplified_mappo/runs/PPO_simple_spread_None_mappo_simple_spread_3agents_noseed_20241124_210343_20241124_210344/config.json"                 # Path to the training config file
    num_episodes: int = 10               # Number of episodes to test
    render: bool = True                  # Whether to render the environment
    delay: float = 0.5                   # Delay between steps for visualization
    deterministic: bool = True           # Whether to use deterministic action selection
    save_video: bool = False             # Whether to save a video of the episodes
    video_path: Optional[str] = None     # Path to save the video
    device: str = "cuda"                  # Device to run the model on 