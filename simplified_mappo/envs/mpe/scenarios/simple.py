from .. import core
from .. import scenario
import numpy as np

class Scenario(scenario.BaseScenario):
    def make_world(self, num_agents=1, num_landmarks=1, episode_length=25):
        world = core.World()
        world.world_length = episode_length
        world.dim_c = 2  # communication channel dimensionality
        world.dim_p = 2  # position dimensionality
        world.num_agents = 1  # force single agent
        world.num_landmarks = 1  # force single landmark
        world.collaborative = False  # single agent, so not collaborative
        world.algorithm = 'mappo'  # Default to MAPPO, can be overridden
        
        # Add single agent
        world.agents = [core.Agent()]
        agent = world.agents[0]
        agent.name = 'agent 0'
        agent.collide = False  # no collision in simple environment
        agent.silent = True
        agent.size = 0.15
        
        # Add single landmark
        world.landmarks = [core.Landmark()]
        landmark = world.landmarks[0]
        landmark.name = 'landmark 0'
        landmark.collide = False
        landmark.movable = False
        
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # Set agent properties
        world.agents[0].color = np.array([0.35, 0.35, 0.85])
        
        # Set landmark properties
        world.landmarks[0].color = np.array([0.25, 0.25, 0.25])
        
        # Random initial position for agent
        world.agents[0].state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        world.agents[0].state.c = np.zeros(world.dim_c)
        
        # Random initial position for landmark
        world.landmarks[0].state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        """
        Compute reward for agent based on whether we're using MAPPO or IPPO
        """
        if world.algorithm == 'mappo':
            return self._mappo_reward(agent, world)
        else:  # IPPO
            return self._ippo_reward(agent, world)

    def _mappo_reward(self, agent, world):
        """Simple MAPPO reward calculation"""
        # Calculate distance to landmark
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        return -dist  # Negative distance as reward

    def _ippo_reward(self, agent, world):
        """More detailed IPPO reward calculation"""
        # Calculate distance to landmark
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        
        # Calculate velocity magnitude
        vel_magnitude = np.sqrt(np.sum(np.square(agent.state.p_vel)))
        
        # Initialize reward
        rew = 0
        
        # Distance-based reward (reduced magnitude)
        rew = -dist * 2.0  # Reduced distance penalty
        
        # Bonus rewards for being close
        if dist < 0.1:
            rew += 2.0  # Reduced bonus
            # Additional reward for being still when close
            if vel_magnitude < 0.1:
                rew += 1.0
        
        # Stronger velocity penalties to discourage overshooting
        vel_penalty = -0.5 * vel_magnitude  # Base velocity penalty
        
        # Even stronger penalty for high velocities
        if vel_magnitude > 0.3:
            vel_penalty *= 2.0
        
        # Add velocity penalty to reward
        rew += vel_penalty
        
        # Penalty for moving away from landmark
        landmark_dir = world.landmarks[0].state.p_pos - agent.state.p_pos
        vel_alignment = np.dot(agent.state.p_vel, landmark_dir)
        if vel_alignment < 0:  # Moving away from landmark
            rew -= 0.5 * abs(vel_alignment)
        
        return rew

    def observation(self, agent, world):
        """
        Returns observation for the agent, which includes:
        - Agent's velocity
        - Agent's position
        - Relative position of landmark
        """
        # Get landmark position relative to agent
        landmark_pos = world.landmarks[0].state.p_pos - agent.state.p_pos
        
        return np.concatenate([
            agent.state.p_vel,           # Agent's velocity
            agent.state.p_pos,           # Agent's absolute position
            landmark_pos                 # Relative position of landmark
        ])
