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
                actions = {}
                
                # Process each agent
                for agent_idx in all_agents:
                    agent_state = state.get(agent_idx, {})
                    agent_index = env.agent_name_to_index[agent_idx]
                    
                    # Get the agent's state array
                    first_agent_state = state[all_agents[0]]
                    
                    try:
                        if isinstance(first_agent_state, dict):
                            agent_state_array = first_agent_state[agent_idx]
                        else:
                            agent_state_array = state[agent_idx]
                            
                        agent_state_tensor = torch.FloatTensor(agent_state_array).to(device)
                        action = ppo_agents[agent_index].select_action(agent_state_tensor)
                    except Exception as e:
                        action = np.zeros(action_dim, dtype=np.float32)
                        
                    actions[agent_idx] = action

                # Step environment
                try:
                    step_result = env.step(actions)
                    
                    if len(step_result) == 4:
                        next_state, rewards, dones, _ = step_result
                    elif len(step_result) == 5:
                        next_state, rewards, dones, _, _ = step_result
                    else:
                        next_state, rewards, dones = step_result[:3]

                    # Check if we got empty dictionaries (episode ended)
                    if not rewards or not dones:
                        break

                    ep_reward += sum(rewards.values())

                    # Add this line to render the environment
                    if test_cfg.render:
                        env.render()
                        
                    if test_cfg.frame_delay > 0:
                        time.sleep(test_cfg.frame_delay)

                    # Handle different done formats
                    if isinstance(dones, dict) and all_agents[0] in dones:
                        episode_done = dones[all_agents[0]]
                    else:
                        episode_done = True

                    if episode_done:
                        break

                    state = next_state

                except Exception as e:
                    break

                # Log step-level metrics
                writer.add_scalar('Test/step_reward', sum(rewards.values()), t + (ep-1)*cfg.env.max_ep_len)

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
    cfg = Config()
    test_cfg = TestConfig()
    test(cfg, test_cfg)
