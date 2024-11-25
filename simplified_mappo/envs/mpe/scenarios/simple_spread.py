from .. import core
from .. import scenario
import numpy as np

class Scenario(scenario.BaseScenario):
    def make_world(self, num_agents=3, num_landmarks=3, episode_length=25):
        world = core.World()
        world.world_length = episode_length
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        world.collaborative = True
        world.algorithm = 'mappo'  # Default to MAPPO, can be overridden
        
        # add agents
        world.agents = [core.Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            
        # add landmarks
        world.landmarks = [core.Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        """
        Compute reward for agent based on whether we're using MAPPO or IPPO
        """
        if world.algorithm == 'mappo':
            return self._mappo_reward(agent, world)
        else:  # IPPO
            return self._ippo_reward(agent, world)

    def _mappo_reward(self, agent, world):
        """Original MAPPO reward calculation"""
        rew = 0
        # Global reward based on minimum distance of any agent to each landmark
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) 
                    for a in world.agents]
            rew -= min(dists)

        # Penalize collisions between agents
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                dist = np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                if dist < 2 * agent.size:
                    rew -= 1
        return rew

    def _ippo_reward(self, agent, world):
        """IPPO-specific reward calculation"""
        rew = 0
        
        # Individual reward based on this agent's distance to its closest landmark
        agent_distances = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) 
                          for l in world.landmarks]
        closest_dist = min(agent_distances)
        rew = -closest_dist  # Base reward
        
        # Add distance threshold bonus
        if closest_dist < 0.1:  # Bonus for being very close
            rew += 1.0
        
        # Small penalty for being too far
        if closest_dist > 1.0:
            rew -= 0.5
            
        # Optional: Very small consideration for other agents' performance
        # to maintain some coordination
        other_agents_reward = 0
        for l in world.landmarks:
            other_dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) 
                          for a in world.agents if a is not agent]
            if other_dists:  # If there are other agents
                other_agents_reward += -min(other_dists) * 0.05  # Small weight for coordination
        
        rew += other_agents_reward

        # Collision penalties (unchanged)
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                dist = np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                if dist < 2 * agent.size:
                    rew -= 1.0

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            
        # get positions of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
