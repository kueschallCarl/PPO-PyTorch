import os
from datetime import datetime
import torch
import numpy as np
from models.ppo import PPO
from utils.wrappers import PettingZooWrapper
from pettingzoo.mpe import simple_v3
from config.config import Config, TestConfig
import time
from dataclasses import dataclass, asdict
from torch.utils.tensorboard import SummaryWriter


def test(cfg: Config, test_cfg: TestConfig):
    print("============================================================================================")
    print(f"Testing started for model: {test_cfg.checkpoint_path}")
    print("============================================================================================")

    # Create env
    raw_env = simple_v3.parallel_env(continuous_actions=cfg.env.continuous_actions, render_mode='human' if test_cfg.render else None)
    first_agent = raw_env.possible_agents[0]
    
    # Get state and action dimensions
    state_dim = raw_env.observation_space(first_agent).shape[0]
    if cfg.env.has_continuous_action_space:
        action_dim = raw_env.action_space(first_agent).shape[0]
    else:
        action_dim = raw_env.action_space(first_agent).n

    env = PettingZooWrapper(raw_env, num_agents=len(raw_env.possible_agents))

    # Set random seed
    if test_cfg.random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", test_cfg.random_seed)
        torch.manual_seed(test_cfg.random_seed)
        env.seed(test_cfg.random_seed)
        np.random.seed(test_cfg.random_seed)

    # Create writer directory path for test results
    writer_dir = os.path.join(cfg.log.tensorboard_dir, 
                           f"TEST_PPO_{cfg.env.env_name}_{test_cfg.random_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(writer_dir)

    # Initialize agents with the writer
    ppo_agents = [
        PPO(state_dim=state_dim,
            action_dim=action_dim,
            cfg=cfg,
            writer=writer)
        for _ in range(len(raw_env.possible_agents))
    ]

    # Load pretrained weights
    print("Loading pretrained model from:", test_cfg.checkpoint_path)
    for agent in ppo_agents:
        agent.load(test_cfg.checkpoint_path)

    # Testing loop
    test_running_reward = 0

    for ep in range(1, test_cfg.total_test_episodes + 1):
        state = env.reset()
        ep_reward = 0
        
        for t in range(1, cfg.env.max_ep_len + 1):
            current_agent = env.current_agent_idx
            action = ppo_agents[current_agent].select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            # Log step-level metrics
            writer.add_scalar('Test/step_reward', reward, t + (ep-1)*cfg.env.max_ep_len)

            if test_cfg.render and test_cfg.frame_delay > 0:
                time.sleep(test_cfg.frame_delay)

            if done:
                break

        # Clear buffers
        for agent in ppo_agents:
            agent.buffer.clear()

        test_running_reward += ep_reward
        print(f'Episode: {ep}/{test_cfg.total_test_episodes} \t Reward: {ep_reward:.2f}')

        # Log episode-level metrics
        writer.add_scalar('Test/episode_reward', ep_reward, ep)
        writer.add_scalar('Test/episode_length', t, ep)
        writer.add_scalar('Test/running_average_reward', test_running_reward/ep, ep)

    env.close()

    # Print and log final summary
    avg_test_reward = test_running_reward / test_cfg.total_test_episodes
    writer.add_scalar('Test/final_average_reward', avg_test_reward, 0)
    
    print("============================================================================================")
    print(f"Average test reward: {avg_test_reward:.2f}")
    print("============================================================================================")

    writer.close()

if __name__ == '__main__':
    # Load configs
    cfg = Config()
    test_cfg = TestConfig()
    
    # You can modify test config here if needed
    # test_cfg.render = False
    # test_cfg.total_test_episodes = 20
    # test_cfg.frame_delay = 0.1
    
    test(cfg, test_cfg)
