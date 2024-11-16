## Value Function Clipping Notes

> so generally speaking for value-clipping: try to take the smaller update, unless the bigger update has a bigger loss
- In the simple env, the loss progression is actually less smooth with clipping.
- At the same time, the logs show that it successfully calms down the changes in value function.
    - mean_value is much smoother and consistent
    - same for mean_value_change and value_std
- needs more testing in more complex environments

### Runs
- at 'testing_ppo_simple_value_clipping_vs_without'


## Explaining Value-Clipping
### Policy Clipping:

The minimum ensures that large advantage values don’t drive overly aggressive changes in the policy.
By clipping, the objective remains stable even if ratio(theta) moves slightly out of bounds, balancing exploration and exploitation.   

### Value Function Clipping:

The maximum ensures that the clipped prediction doesn't make the value loss artificially small.
By taking the larger loss, the value function is encouraged to improve when it is far from the target (returns), while still restricting large updates.

| **Aspect**                  | **Policy Clipping**                                   | **Value Function Clipping**                         |
|-----------------------------|-----------------------------------------------------|---------------------------------------------------|
| **What is clipped?**        | Probability ratio \(r_t(\theta)\).                  | Value function prediction \(V(s_t; \theta)\).     |
| **Why clip?**               | To constrain policy updates and ensure stability.   | To prevent overfitting and large value changes.   |
| **Objective uses...**       | Minimum (clipped vs. unclipped).                    | Maximum (clipped vs. unclipped loss).             |
| **Main Goal**               | Stability and trust-region updates for policy.      | Stability and controlled updates for value loss.  |
| **Mechanism Protects Against** | Overly aggressive policy changes.                   | Overfitting to noisy returns or unstable updates. |

