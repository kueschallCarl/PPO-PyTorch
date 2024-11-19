import gym

class PettingZooWrapper:
    def __init__(self, env, num_agents):
        self.env = env
        self.num_agents = num_agents
        self.agent_name_to_index = {agent: idx for idx, agent in enumerate(env.possible_agents)}
        self._current_agent_idx = 0

    def reset(self):
        obs_tuple = self.env.reset()
        # Convert tuple to dict if necessary
        if isinstance(obs_tuple, tuple):
            return dict(zip(self.env.possible_agents, obs_tuple))
        return obs_tuple

    def step(self, actions):
        try:
            result = self.env.step(actions)
            
            # Handle different return formats
            if len(result) == 5:
                next_state, rewards, dones, infos, _ = result
            else:
                next_state, rewards, dones, infos = result
            
            # Convert next_state tuple to dict if necessary
            if isinstance(next_state, tuple):
                next_state = dict(zip(self.env.possible_agents, next_state))
            
            # Convert dones to a single boolean if it's a dict
            if isinstance(dones, dict):
                done = all(dones.values())
            else:
                done = dones
            
            return next_state, rewards, dones, infos
            
        except Exception as e:
            print(f"Error in wrapper step: {e}")
            print(f"Actions provided: {actions}")
            print(f"Step result: {result}")
            raise

    def close(self):
        self.env.close()

    @property
    def current_agent_idx(self):
        current_agent = self.env.agent_selection
        return self.agent_name_to_index[current_agent]

    def seed(self, seed):
        self.env.seed(seed)

    # ... rest of PettingZooWrapper class methods ... 