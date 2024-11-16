# PPO_MARL_doc

## Evaluating MPE Environments with PPO
Setup:
- All agents share the same model
### Training
#### Hyperparameters
- total_timesteps: int = 500000
- checkpoint_freq: int = 10000
- learning_rate: float = 3e-4
- n_steps: int = 2048
- batch_size: int = 64
- n_epochs: int = 10
- gamma: float = 0.99
- policy_kwargs: dict = None

### Evaluating Performance

#### Adversary
1. Metrics:
Final Training Metrics:
```
-----------------------------------------
| time/                   |             |
|    fps                  | 2231        |
|    iterations           | 41          |
|    time_elapsed         | 225         |
|    total_timesteps      | 503808      |
| train/                  |             |
|    approx_kl            | 0.003818611 |
|    clip_fraction        | 0.0364      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.88       |
|    explained_variance   | 0.8490455   |
|    learning_rate        | 0.0003      |
|    loss                 | 2.78        |
|    n_updates            | 400         |
|    policy_gradient_loss | -0.00246    |
|    value_loss           | 4.58        |
-----------------------------------------
```
Final Evaluation Metrics:
```
Episode 10/10 completed

Evaluation Results:
--------------------------------------------------
| Metric      |     Value |
|:------------|----------:|
| reward/mean | -0.638555 |
| reward/std  |  1.98327  |
| reward/min  | -3.82283  |
| reward/max  |  2.81661  |
| length/mean | 25        |
| length/std  |  0        |
| length/min  | 25        |
| length/max  | 25        |
```
2. Interpreting Results:
    - Agents run towards goal landmark without much strategy but consistently succeed
    - They are however **not** able to divert the adversary from the goal landmark
#### Spread
1. Metrics:
Final Training Metrics:
```
------------------------------------------
| time/                   |              |
|    fps                  | 2127         |
|    iterations           | 41           |
|    time_elapsed         | 236          |
|    total_timesteps      | 503808       |
| train/                  |              |
|    approx_kl            | 0.0067156386 |
|    clip_fraction        | 0.0707       |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.17        |
|    explained_variance   | 0.6572335    |
|    learning_rate        | 0.0003       |
|    loss                 | 3.7          |
|    n_updates            | 400          |
|    policy_gradient_loss | -0.00625     |
|    value_loss           | 7.54         |
------------------------------------------
```
Final Evaluation Metrics:
```
Episode 10/10 completed

Evaluation Results:
--------------------------------------------------
| Metric      |    Value |
|:------------|---------:|
| reward/mean | -20.0477 |
| reward/std  |   4.0012 |
| reward/min  | -28.6228 |
| reward/max  | -13.4795 |
| length/mean |  25      |
| length/std  |   0      |
| length/min  |  25      |
| length/max  |  25      |
```
2. Interpreting Results:
    - Lazy-Agent-Problem
        - Sometimes one of the agents does all the work will others wait
    - No communication possible
    - They all move towards the landmarks in one big mass as you'd expect

#### Reference
1. Metrics:
Final Training Metrics:
```
-----------------------------------------
| time/                   |             |
|    fps                  | 1966        |
|    iterations           | 62          |
|    time_elapsed         | 258         |
|    total_timesteps      | 507904      |
| train/                  |             |
|    approx_kl            | 0.014593273 |
|    clip_fraction        | 0.169       |
|    clip_range           | 0.2         |
|    entropy_loss         | -3.02       |
|    explained_variance   | 0.59992766  |
|    learning_rate        | 0.0003      |
|    loss                 | 5.1         |
|    n_updates            | 610         |
|    policy_gradient_loss | -0.0317     |
|    value_loss           | 12.2        |
-----------------------------------------
```
Final Evaluation Metrics:
```
Episode 10/10 completed

Evaluation Results:
--------------------------------------------------
| Metric      |     Value |
|:------------|----------:|
| reward/mean | -17.1694  |
| reward/std  |   6.29205 |
| reward/min  | -30.723   |
| reward/max  |  -8.59374 |
| length/mean |  25       |
| length/std  |   0       |
| length/min  |  25       |
| length/max  |  25       |
```
2. Interpreting Results:
    - No communication possible so each agent cannot tell its partner where their goal is. 
    - Since the agents share networks but get different observations the trajectory averages out to move somewhere toward the cluster of landmarks


#### Simple
1. Metrics:
Final Training Metrics:
```
-----------------------------------------
| time/                   |             |
|    fps                  | 1461        |
|    iterations           | 123         |
|    time_elapsed         | 344         |
|    total_timesteps      | 503808      |
| train/                  |             |
|    approx_kl            | 0.003859388 |
|    clip_fraction        | 0.0462      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.421      |
|    explained_variance   | 0.999033    |
|    learning_rate        | 0.0003      |
|    loss                 | -0.000944   |
|    n_updates            | 1220        |
|    policy_gradient_loss | -0.00184    |
|    value_loss           | 0.00823     |
-----------------------------------------
```
Final Evaluation Metrics:
```
Episode 10/10 completed

Evaluation Results:
--------------------------------------------------
| Metric      |      Value |
|:------------|-----------:|
| reward/mean | -10.2039   |
| reward/std  |   9.0897   |
| reward/min  | -30.4457   |
| reward/max  |  -0.295281 |
| length/mean |  25        |
| length/std  |   0        |
| length/min  |  25        |
| length/max  |  25        |
```
2. Interpreting Results:
    - Not much to say the agent moves towards the goal landmark and that's it.
    - PPO can handle this task with ease.

#### Speaker-Listener
1. Metrics:
Final Training Metrics:
```
------------------------------------------
| time/                   |              |
|    fps                  | 1963         |
|    iterations           | 62           |
|    time_elapsed         | 258          |
|    total_timesteps      | 507904       |
| train/                  |              |
|    approx_kl            | 0.0048693176 |
|    clip_fraction        | 0.0642       |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.885       |
|    explained_variance   | 0.30216777   |
|    learning_rate        | 0.0003       |
|    loss                 | 7.11         |
|    n_updates            | 610          |
|    policy_gradient_loss | -0.00309     |
|    value_loss           | 16.4         |
------------------------------------------
```
Final Evaluation Metrics:
```
Episode 10/10 completed

Evaluation Results:
--------------------------------------------------
| Metric      |     Value |
|:------------|----------:|
| reward/mean | -17.7612  |
| reward/std  |  12.9432  |
| reward/min  | -45.5213  |
| reward/max  |  -5.52822 |
| length/mean |  25       |
| length/std  |   0       |
| length/min  |  25       |
| length/max  |  25       |
```
2. Interpreting Results:
    - Obviously without communication the agents do not perform well
    - The listener doesn't actually get any observation of the goal landmark from the speaker. The best it can do is to move towards the center of the environment.




