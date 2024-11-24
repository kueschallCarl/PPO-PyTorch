import json
import torch
import numpy as np
from algorithms.ppo import PPO
from algorithms.mappo import MAPPO
from algorithms.actor_critic import MLPPolicy
from config.config import Config
from config.test_config import TestConfig
import argparse
from envs.mpe.scenarios import SCENARIOS
import time
import matplotlib.pyplot as plt
from dataclasses import asdict
import os

def get_absolute_path(relative_path: str) -> str:
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Convert relative path to absolute path
    return os.path.abspath(os.path.join(script_dir, '..', relative_path))

def make_env(cfg, render_mode=None):
    """
    Create a Multi-Agent Particle Environment (MPE) using local implementation
    """
    # Get the scenario class from the scenarios
    scenario = SCENARIOS[cfg.env.env_name]()
    
    # Create world
    world = scenario.make_world(
        num_agents=cfg.env.num_agents,
        num_landmarks=cfg.env.num_agents if cfg.env.env_name == "simple_spread" else None,
        episode_length=cfg.env.episode_length
    )
    
    # Get dimensions
    obs_dim = len(scenario.observation(world.agents[0], world))
    if cfg.env.has_continuous_action_space:
        action_dim = world.dim_p  # Physical action dimension
    else:
        action_dim = 5  # Example: 5 discrete actions
    
    return world, obs_dim, action_dim, scenario

def test_model(train_cfg: Config, test_cfg: TestConfig):
    """
    Test a trained model
    
    Args:
        train_cfg: Training configuration
        test_cfg: Testing configuration
    """
    # Initialize environment
    world, state_dim, action_dim, scenario = make_env(train_cfg, render_mode='human' if test_cfg.render else None)
    device = torch.device(test_cfg.device)
    
    # Initialize policy based on algorithm
    if train_cfg.algorithm == "mappo":
        policy = MLPPolicy(
            obs_space=state_dim,
            action_space=action_dim
        ).to(device)
        model_path = get_absolute_path(test_cfg.model_path)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        
    elif train_cfg.algorithm == "ippo":
        policies = []
        for i in range(train_cfg.env.num_agents):
            agent = PPO(
                state_dim=state_dim,
                action_dim=action_dim,
                cfg=train_cfg
            )
            # Get absolute path for base path
            base_path = get_absolute_path(test_cfg.model_path.replace('_agent0.pth', ''))
            agent_path = f"{base_path}_agent{i}.pth"
            print(f"Loading model from: {agent_path}")  # Debug print
            agent.load(agent_path)
            policies.append(agent)
    
    print(f"Testing {train_cfg.algorithm} on {train_cfg.env.env_name} environment")
    print(f"Loaded model from: {test_cfg.model_path}")
    
    # Test loop
    episode_rewards = []
    for episode in range(test_cfg.num_episodes):
        scenario.reset_world(world)
        episode_reward = 0
        
        for step in range(train_cfg.env.episode_length):
            if train_cfg.algorithm == "mappo":
                # Get observations for all agents
                obs = torch.FloatTensor(np.array([
                    scenario.observation(agent, world) for agent in world.agents
                ])).to(device)
                
                # Get actions from policy
                with torch.no_grad():
                    actions, _ = policy.get_actions(obs)
                actions = actions.cpu().numpy()
                
                # Set actions for each agent
                for agent, action in zip(world.agents, actions):
                    agent.action.u = action
                    
            elif train_cfg.algorithm == "ippo":
                # Get actions from individual policies
                for i, agent in enumerate(world.agents):
                    obs = scenario.observation(agent, world)
                    obs_tensor = torch.FloatTensor(obs).to(device)
                    action = policies[i].select_action(obs_tensor, deterministic=test_cfg.deterministic)
                    agent.action.u = action
            
            # Step the environment
            world.step()
            
            # Calculate rewards
            rewards = [scenario.reward(agent, world) for agent in world.agents]
            episode_reward += sum(rewards) / len(rewards)
            
            if test_cfg.render:
                render_env(world)
                time.sleep(test_cfg.delay)
                
        avg_reward = episode_reward / train_cfg.env.episode_length
        episode_rewards.append(avg_reward)
        print(f"Episode {episode + 1}/{test_cfg.num_episodes}, Average Reward: {avg_reward:.2f}")
    
    # Print final statistics
    print("\nTest Results:")
    print(f"Average Reward over {test_cfg.num_episodes} episodes: {np.mean(episode_rewards):.2f}")
    print(f"Std Dev of Rewards: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")

def render_env(world):
    """
    Render the environment using matplotlib
    """
    plt.clf()
    
    # Plot landmarks
    landmark_pos = np.array([l.state.p_pos for l in world.landmarks])
    plt.scatter(landmark_pos[:, 0], landmark_pos[:, 1], c='gray', s=100, label='Landmarks')
    
    # Plot agents
    agent_pos = np.array([a.state.p_pos for a in world.agents])
    plt.scatter(agent_pos[:, 0], agent_pos[:, 1], c='blue', s=200, label='Agents')
    
    # Add velocity arrows
    for agent in world.agents:
        plt.arrow(agent.state.p_pos[0], agent.state.p_pos[1],
                 agent.state.p_vel[0]*0.1, agent.state.p_vel[1]*0.1,
                 head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)

def main():
    parser = argparse.ArgumentParser()
    test_cfg = TestConfig()
    
    # Add arguments for all TestConfig fields, using the defaults from TestConfig
    parser.add_argument("--model_path", type=str, default=test_cfg.model_path,
                       help="Path to the saved model")
    parser.add_argument("--config_path", type=str, default=test_cfg.config_path,
                       help="Path to the config file")
    parser.add_argument("--num_episodes", type=int, default=test_cfg.num_episodes,
                       help="Number of episodes to test")
    parser.add_argument("--render", action="store_true", default=test_cfg.render,
                       help="Render the environment")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    parser.add_argument("--delay", type=float, default=test_cfg.delay,
                       help="Delay between steps for visualization")
    parser.add_argument("--deterministic", action="store_true", default=test_cfg.deterministic,
                       help="Use deterministic action selection")
    parser.add_argument("--save_video", action="store_true", default=test_cfg.save_video,
                       help="Save video of episodes")
    parser.add_argument("--video_path", type=str, default=test_cfg.video_path,
                       help="Path to save video")
    parser.add_argument("--device", type=str, default=test_cfg.device,
                       help="Device to run the model on")
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths if needed
    config_path = get_absolute_path(args.config_path)
    model_path = get_absolute_path(args.model_path)
    
    # Load and parse config
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
        
    # Create Config object from the nested 'config' dictionary
    train_cfg = Config()
    for category, settings in loaded_config['config'].items():
        if hasattr(train_cfg, category):
            category_config = getattr(train_cfg, category)
            if isinstance(settings, dict):
                for key, value in settings.items():
                    if hasattr(category_config, key):
                        setattr(category_config, key, value)
            else:
                setattr(train_cfg, category, settings)
    
    # Update test config with CLI arguments
    for key, value in vars(args).items():
        if hasattr(test_cfg, key):
            setattr(test_cfg, key, value)
    
    # Handle special case for render flag if it exists
    if hasattr(args, 'no_render') and args.no_render:
        test_cfg.render = False
    
    test_model(train_cfg, test_cfg)

if __name__ == "__main__":
    main() 