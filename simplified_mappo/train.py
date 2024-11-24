import torch
from algorithms.actor_critic import MLPPolicy
from algorithms.mappo import MAPPO
from utils.buffer import SharedReplayBuffer
from runner import Runner
import logging
import time
from datetime import datetime
import traceback

def train(
    scenario_name="simple_spread",
    num_agents=3,
    num_episodes=1000,
    episode_length=25,
    buffer_size=2048,
    batch_size=64,
    learning_rate=3e-4,
    eval_frequency=100
):
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
        logging.info(f"Scenario: {scenario_name}, Agents: {num_agents}")
        logging.info(f"Episodes: {num_episodes}, Episode length: {episode_length}")
        logging.info(f"Buffer size: {buffer_size}, Batch size: {batch_size}, LR: {learning_rate}")

        # Initialize device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        # Initialize environment and runner
        runner = Runner(scenario_name, num_agents, device=device)
        
        # Initialize policy and algorithm
        policy = MLPPolicy(
            obs_space=runner.obs_space,
            action_space=runner.action_space
        ).to(device)
        
        algorithm = MAPPO(
            policy=policy,
            lr=learning_rate
        )
        
        # Initialize buffer
        buffer = SharedReplayBuffer(
            num_agents=num_agents,
            obs_space=runner.obs_space,
            act_space=runner.action_space,
            size=buffer_size,
            device=device
        )
        
        # Training loop
        start_time = time.time()
        for episode in range(num_episodes):
            episode_start = time.time()
            
            try:
                # Collect experience
                episode_reward = runner.collect_episodes(policy, buffer, episode_length)
                
                # Update policy
                for sample in buffer.get_samples(batch_size):
                    algorithm.update(sample)
                
                # Log progress
                episode_duration = time.time() - episode_start
                if (episode + 1) % 10 == 0:
                    logging.info(f"Episode {episode + 1}/{num_episodes} completed in {episode_duration:.2f}s. "
                               f"Reward: {episode_reward:.2f}")
                
                # Evaluate policy
                if (episode + 1) % eval_frequency == 0:
                    logging.info("Starting evaluation...")
                    try:
                        eval_reward = runner.eval_policy(policy)
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
        logging.info(f"Training completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Fatal error during training: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="simple_spread", 
                       choices=["simple_spread"])
    parser.add_argument("--num_agents", type=int, default=3)
    args = parser.parse_args()
    
    train(scenario_name=args.scenario, num_agents=args.num_agents) 