# PPO GAE no value clipping Hyperparameter Optimization Notes

## Best Params (Trial 22)
### Hyperparameters
- lr_actor: 0.000188176682200799
- lr_critic: 0.0003559365444027006
- gamma: 0.9068490977704631
- gae_lambda: 0.9416398917943807
- K_epochs: 81
- eps_clip: 0.1256265613656234
- action_std: 0.7770917843215948
- min_action_std: 0.1434606566175461

### Metadata
- "best_value": -0.6095328338309707,
- "study_name": "ppo_optimization",
- "n_trials": 80,
- "timestamp": "2024-11-17 00:51:13"



## Notes

### Trials of Note
- Horrible performance
    1. PPO_simple_v3_0_0_tune_trial_13_20241116_231819
    2. PPO_simple_v3_0_0_tune_trial_49_20241117_000742
- Good performance
    1. PPO_simple_v3_0_0_tune_trial_21_20241116_232910
- Best performing trial
    1. PPO_simple_v3_0_0_tune_trial_74_20241117_004028 -> According to Loss
    2. PPO_simple_v3_0_0_tune_trial_22_20241116_233032 -> According to Rolling Average Reward (Optuna)

