import gym
from gym.spaces import Box
import numpy as np
from .mpe.scenarios import SCENARIOS

class MPEEnv(gym.Env):
    def __init__(self, scenario_name, num_agents, episode_length=25):
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Scenario {scenario_name} not found. Available scenarios: {list(SCENARIOS.keys())}")
            
        self.scenario = SCENARIOS[scenario_name]()
        scenario_params = {
            'num_agents': num_agents,
            'episode_length': episode_length
        }
        
        if scenario_name == 'simple_spread':
            scenario_params['num_landmarks'] = num_agents
            
        self.world = self.scenario.make_world(**scenario_params)
        
        # configure spaces
        self.action_space = [Box(low=-1, high=1, shape=(2,)) for _ in range(num_agents)]
        obs_dim = len(self.scenario.observation(self.world.agents[0], self.world))
        self.observation_space = [Box(low=-np.inf, high=+np.inf, shape=(obs_dim,)) 
                                for _ in range(num_agents)]
        
    def reset(self):
        self.scenario.reset_world(self.world)
        obs = [self.scenario.observation(agent, self.world) for agent in self.world.agents]
        return obs
        
    def step(self, actions):
        # set actions for each agent
        for agent, action in zip(self.world.agents, actions):
            agent.action.u = action
            
        # advance world state
        self.world.step()
        
        # record observation for each agent
        obs = [self.scenario.observation(agent, self.world) for agent in self.world.agents]
        rewards = [self.scenario.reward(agent, self.world) for agent in self.world.agents]
        dones = [False] * len(self.world.agents)  # Simple spread doesn't have a natural termination
        
        return obs, rewards, dones, {}

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed) 