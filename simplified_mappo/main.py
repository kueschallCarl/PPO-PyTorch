import argparse
from config.config import Config
from train_ippo import train_ippo
from train import train_mappo

def main():
    parser = argparse.ArgumentParser()
    
    # Basic arguments
    parser.add_argument("--algorithm", type=str, default="ippo", choices=["ippo", "mappo"])
    parser.add_argument("--env.env_name", type=str, default="simple_spread")
    parser.add_argument("--env.num_agents", type=int)
    parser.add_argument("--env.episode_length", type=int)
    parser.add_argument("--seed", type=int)
    
    # Logging arguments
    parser.add_argument("--log.wandb_project", type=str)
    parser.add_argument("--log.run_name", type=str)
    parser.add_argument("--log.use_wandb", type=bool)
    
    # Training arguments
    parser.add_argument("--training.lr_actor", type=float)
    parser.add_argument("--training.lr_critic", type=float)
    
    args = parser.parse_args()
    cfg = Config.from_args(args)
    
    # Route to appropriate training function
    if cfg.algorithm == "ippo":
        train_ippo(cfg)
    else:
        train_mappo(cfg)

if __name__ == "__main__":
    main() 