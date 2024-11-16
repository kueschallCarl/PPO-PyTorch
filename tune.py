import optuna
from train import train
from config.config import Config
import torch
import numpy as np
from datetime import datetime
import os
import json
from dataclasses import asdict

def objective(trial):
    """
    Objective function for Optuna to optimize.
    Returns the average reward over training episodes.
    """
    # Create a config with trial-suggested hyperparameters
    cfg = Config()
    
    # PPO hyperparameters to tune
    cfg.ppo.lr_actor = trial.suggest_float('lr_actor', 1e-5, 1e-3, log=True)
    cfg.ppo.lr_critic = trial.suggest_float('lr_critic', 1e-5, 1e-3, log=True)
    cfg.ppo.gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    cfg.ppo.gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    cfg.ppo.K_epochs = trial.suggest_int('K_epochs', 5, 100)
    cfg.ppo.eps_clip = trial.suggest_float('eps_clip', 0.1, 0.3)
    
    # Action space exploration parameters
    cfg.action.action_std = trial.suggest_float('action_std', 0.1, 1.0)
    cfg.action.min_action_std = trial.suggest_float('min_action_std', 0.01, 0.2)
    
    # Modify training parameters for faster iteration during tuning
    cfg.env.max_training_timesteps = int(1e5)  # Reduced training time for tuning
    cfg.log.save_model_freq = int(5e4)
    
    # Set unique run name for this trial
    cfg.log.run_name = f"tune_trial_{trial.number}"
    
    # Train with these hyperparameters
    try:
        final_reward = train(cfg, return_reward=True)
        
        # Save trial results
        trial_dir = os.path.join("tuning_results", f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Save configuration and results
        result = {
            "trial_number": trial.number,
            "final_reward": float(final_reward),
            "params": trial.params,
            "config": asdict(cfg)
        }
        
        with open(os.path.join(trial_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=4)
            
        return final_reward
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        return float('-inf')

def tune_hyperparameters(n_trials=50, study_name="ppo_optimization"):
    """
    Run hyperparameter tuning with Optuna.
    
    Args:
        n_trials (int): Number of trials to run
        study_name (str): Name of the study for saving results
    """
    # Create study directory
    study_dir = os.path.join("tuning_results", study_name)
    os.makedirs(study_dir, exist_ok=True)
    
    # Create and run study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{study_dir}/study.db",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "study_name": study_name,
        "n_trials": n_trials,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(study_dir, "best_params.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Print results
    print("\nBest trial:")
    print(f"Value: {best_value}")
    print("Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    
    # Create plots
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig2 = optuna.visualization.plot_param_importances(study)
        
        # Save plots
        fig1.write_html(os.path.join(study_dir, "optimization_history.html"))
        fig2.write_html(os.path.join(study_dir, "param_importances.html"))
    except:
        print("Warning: Could not create visualization plots")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    tune_hyperparameters(n_trials=2) 