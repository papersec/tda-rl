class ExplorationStrategy(object):
    def __init__(self, config):
        pass
    
    def choose_action(self, timestep, network, environment):
        raise NotImplementedError