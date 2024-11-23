import os
from datetime import datetime
import torch
import numpy as np
from models.ppo import PPO
from config.config import Config, TestConfig
import time
from torch.utils.tensorboard import SummaryWriter
from utils.env_factory import make_env

def test(cfg: Config, test_cfg: TestConfig):
    print("=" * 92)
    print(f"Testing started for model: {test_cfg.checkpoint_path}")
    print("=" * 92)

    # Create env using factory with render mode
    env, state_dim, action_dim = make_env(cfg, render_mode='human' if test_cfg.render else None)

    # Set random seed
    if test_cfg.random_seed:
        print("-" * 92)
        print("Setting random seed to ", test_cfg.random_seed)
        torch.manual_seed(test_cfg.random_seed)
        env.seed(test_cfg.random_seed)
        np.random.seed(test_cfg.random_seed)

    # Create writer directory path for test results
    writer_dir = os.path.join(cfg.log.tensorboard_dir, 
                           f"TEST_PPO_{cfg.env.env_name}_{test_cfg.random_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(writer_dir)

    # Initialize agents without writer (since we're testing)
    device = torch.device(cfg.device)
    ppo_agents = [
        PPO(state_dim=state_dim,
            action_dim=action_dim,
            cfg=cfg)
        for _ in range(env.num_agents)
    ]

    # Load pretrained weights
    print("Loading pretrained model from:", test_cfg.checkpoint_path)
    for agent in ppo_agents:
        agent.load(test_cfg.checkpoint_path)
        agent.policy_old.eval()  # Set policy network to evaluation mode
        agent.policy.eval()      # Set policy network to evaluation mode

    # Testing loop
    test_running_reward = 0
    
    with torch.no_grad():  # Disable gradient computation
        for ep in range(1, test_cfg.total_test_episodes + 1):
            # Update reset() call to handle tuple return
            state_tuple = env.reset()
            observations = state_tuple[0]  # First element contains observations
            ep_reward = 0
            t = 0
            
            if test_cfg.render:
                env.render()  # Render initial state
                time.sleep(test_cfg.frame_delay)
            
            while True:
                actions = {}
                
                # Process each agent
                for agent_idx, obs in observations.items():
                    agent_index = env.agent_name_to_index[agent_idx]
                    
                    # Convert observation to tensor
                    agent_state_tensor = torch.FloatTensor(obs).to(device)
                    
                    # Get action from policy
                    action = ppo_agents[agent_index].select_action(agent_state_tensor, deterministic=True)                    
                    # Ensure action is in the correct format
                    if cfg.env.has_continuous_action_space:
                        action = action.flatten()
                    else:
                        action = int(action)
                    
                    actions[agent_idx] = action

                # Step environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Sum rewards for all agents
                ep_reward += sum(rewards.values())
                t += 1

                if test_cfg.render:
                    env.render()
                    time.sleep(test_cfg.frame_delay)

                # Check if all agents are done
                if all(terminations.values()) or all(truncations.values()):
                    print("Episode finished")
                    break

                # Optional: Limit episode length
                if t >= cfg.env.max_ep_len:
                    print("Max episode length reached")
                    break

                # Log step-level metrics
                writer.add_scalar('Test/step_reward', sum(rewards.values()), t + (ep - 1) * cfg.env.max_ep_len)

            test_running_reward += ep_reward
            print(f'Episode: {ep}/{test_cfg.total_test_episodes} \t Reward: {ep_reward:.2f}')

            # Log episode-level metrics
            writer.add_scalar('Test/episode_reward', ep_reward, ep)
            writer.add_scalar('Test/episode_length', t, ep)
            writer.add_scalar('Test/running_average_reward', test_running_reward / ep, ep)

    env.close()

    # Print and log final summary
    avg_test_reward = test_running_reward / test_cfg.total_test_episodes
    writer.add_scalar('Test/final_average_reward', avg_test_reward, 0)
    
    print("=" * 92)
    print(f"Average test reward: {avg_test_reward:.2f}")
    print("=" * 92)

    writer.close()

if __name__ == '__main__':
    cfg = Config()
    test_cfg = TestConfig()
    test(cfg, test_cfg)
