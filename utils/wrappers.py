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
            
            if len(result) == 5:
                next_state, rewards, dones, infos, _ = result
            else:
                next_state, rewards, dones, infos = result
            
            # Convert next_state tuple to dict if necessary
            if isinstance(next_state, tuple):
                next_state = dict(zip(self.env.possible_agents, next_state))
            
            # Ensure dones is a dictionary with agent names as keys
            if isinstance(dones, bool):
                dones = {agent: dones for agent in self.env.possible_agents}
            
            return next_state, rewards, dones, infos
            
        except Exception as e:
            print(f"Error in wrapper step: {e}")
            print(f"Actions provided: {actions}")
            raise

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

    # ... rest of PettingZooWrapper class methods ... 