import numpy as np

# physical/external base state of all entities
class EntityState(object):
    def __init__(self):
        self.p_pos = None
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super().__init__()
        self.c = None

class Action(object):
    def __init__(self):
        self.u = None
        self.c = None

class Entity(object):
    def __init__(self):
        self.name = ''
        self.size = 0.050
        self.state = EntityState()
        self.state.p_pos = np.zeros(2)
        self.state.p_vel = np.zeros(2)
        self.color = None
        self.movable = False

class Agent(Entity):
    def __init__(self):
        super().__init__()
        self.state = AgentState()
        self.action = Action()
        self.state.c = np.zeros(2)
        self.action.u = np.zeros(2)
        self.action.c = np.zeros(2)
        self.collide = True
        self.silent = False

class Landmark(Entity):
    def __init__(self):
        super().__init__()

class World(object):
    def __init__(self):
        self.agents = []
        self.landmarks = []
        self.dim_c = 0
        self.dim_p = 2
        self.collaborative = False
        self.world_length = 25
        
    def step(self):
        # update agents
        for agent in self.agents:
            # physical action
            agent.state.p_vel = agent.action.u
            agent.state.p_pos += agent.state.p_vel

    def get_obs(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos) 