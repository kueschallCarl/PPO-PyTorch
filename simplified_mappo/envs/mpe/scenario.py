class BaseScenario(object):
    def make_world(self, **kwargs):
        raise NotImplementedError()

    def reset_world(self, world):
        raise NotImplementedError()

    def reward(self, agent, world):
        raise NotImplementedError()

    def observation(self, agent, world):
        raise NotImplementedError() 