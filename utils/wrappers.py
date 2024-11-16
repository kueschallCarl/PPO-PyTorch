import gym

class PettingZooWrapper:
    def __init__(self, env_name, num_agents):
        self.env = env_name
        self.num_agents = num_agents
        self.agents = self.env.possible_agents
        self.reset()
        
    def reset(self):
        obs = self.env.reset()
        self.current_agent_idx = 0
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs[self.agents[0]]
        
    def step(self, action):
        agent = self.agents[self.current_agent_idx]
        actions = {a: action if a == agent else None for a in self.agents}
        
        step_result = self.env.step(actions)
        obs, rewards, dones, truncated, infos = step_result
        
        if len(obs) == 0 or len(rewards) == 0 or len(dones) == 0:
            return self.reset(), 0.0, True, {}
            
        current_obs = obs[agent]
        current_reward = rewards[agent]
        current_done = dones[agent]
        current_info = infos[agent] if agent in infos else {}
        
        self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents
        
        return current_obs, current_reward, current_done, current_info
        
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()

    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        
    # ... rest of PettingZooWrapper class methods ... 