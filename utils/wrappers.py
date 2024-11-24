import gym

class PettingZooWrapper:
    """Wrapper for PettingZoo environments to standardize the interface"""
    
    def __init__(self, env, num_agents):
        self.env = env
        self.num_agents = num_agents
        self.agent_name_to_index = {agent: idx for idx, agent in enumerate(env.possible_agents)}
        
    def reset(self, seed=None):
        """Reset the environment with optional seed"""
        return self.env.reset(seed=seed)
    
    def step(self, actions):
        """Take a step in the environment"""
        return self.env.step(actions)
    
    def close(self):
        """Close the environment"""
        self.env.close()
        
    def render(self):
        """Render the environment"""
        return self.env.render()
