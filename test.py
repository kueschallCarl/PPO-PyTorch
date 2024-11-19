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
    print("============================================================================================")
    print(f"Testing started for model: {test_cfg.checkpoint_path}")
    print("============================================================================================")

    # Create env using factory
    env, state_dim, action_dim = make_env(cfg)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppo_agents = [
        PPO(state_dim=state_dim,
            action_dim=action_dim,
            cfg=cfg,
            writer=writer)
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
            state = env.reset()
            ep_reward = 0
            
            # Get list of all agents
            all_agents = list(env.agent_name_to_index.keys())
            
            for t in range(1, cfg.env.max_ep_len + 1):
                # Initialize actions dictionary
                actions = {}
                
                # Process each agent
                for agent_idx in all_agents:
                    agent_index = env.agent_name_to_index[agent_idx]
                    
                    # Get the agent's state array
                    first_agent_state = state[all_agents[0]]  # Use the first agent's state dict
                    if isinstance(first_agent_state, dict) and agent_idx in first_agent_state:
                        agent_state_array = first_agent_state[agent_idx]
                    else:
                        # Direct state access if not nested
                        agent_state_array = state[agent_idx]
                        
                    agent_state_tensor = torch.FloatTensor(agent_state_array).to(device)
                    action = ppo_agents[agent_index].select_action(agent_state_tensor)
                    actions[agent_idx] = action

                # Step environment
                next_state, rewards, dones, _ = env.step(actions)
                ep_reward += sum(rewards.values())

                # Log step-level metrics
                writer.add_scalar('Test/step_reward', sum(rewards.values()), t + (ep-1)*cfg.env.max_ep_len)

                if test_cfg.render and test_cfg.frame_delay > 0:
                    time.sleep(test_cfg.frame_delay)

                if dones[all_agents[0]]:  # Check first agent's done status
                    break

                state = next_state

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
