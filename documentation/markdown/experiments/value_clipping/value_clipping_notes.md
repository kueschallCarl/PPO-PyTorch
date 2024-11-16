## Value Function Clipping Notes
- In the simple env, the loss progression is actually less smooth with clipping.
- At the same time, the logs show that it successfully calms down the changes in value function.
    - mean_value is much smoother and consistent
    - same for mean_value_change and value_std
- needs more testing in more complex environments

### Runs
- at 'testing_ppo_simple_value_clipping_vs_without'