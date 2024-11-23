from typing import Tuple
from pettingzoo.mpe import simple_v3, simple_adversary_v3, simple_spread_v3, simple_speaker_listener_v4
from utils.wrappers import PettingZooWrapper
from config.config import Config
import numpy as np

# Dictionary mapping environment names to their constructor functions
MPE_ENVS = {
    "simple_v3": simple_v3,
    "simple_adversary_v3": simple_adversary_v3,
    "simple_spread_v3": simple_spread_v3,
    "simple_speaker_listener_v3": simple_speaker_listener_v4
}

def make_env(cfg: Config, render_mode: str = None) -> Tuple[PettingZooWrapper, int, int]:
    """
    Creates and wraps a PettingZoo MPE environment based on config settings.
    
    Args:
        cfg: Configuration object containing environment settings
        render_mode: Rendering mode for the environment ('human', 'rgb_array', etc.)
        
    Returns:
        env: Wrapped environment
        state_dim: Dimension of state space
        action_dim: Dimension of action space
    """
    if cfg.env.env_name not in MPE_ENVS:
        raise ValueError(f"Environment {cfg.env.env_name} not found. Available environments: {list(MPE_ENVS.keys())}")
    
    # Create raw environment
    env_constructor = MPE_ENVS[cfg.env.env_name]
    raw_env = env_constructor.parallel_env(
        continuous_actions=cfg.env.continuous_actions,
        render_mode=render_mode,
        max_cycles=cfg.env.max_ep_len
    )
    
    # Get first agent for space dimensions
    first_agent = raw_env.possible_agents[0]
    
    # Get state and action dimensions
    state_dim = raw_env.observation_space(first_agent).shape[0]
    if cfg.env.has_continuous_action_space:
        action_dim = raw_env.action_space(first_agent).shape[0]
    else:
        action_dim = raw_env.action_space(first_agent).n

    # Wrap environment
    env = PettingZooWrapper(raw_env, num_agents=len(raw_env.possible_agents))
    
    return env, state_dim, action_dim 