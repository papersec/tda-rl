import numpy as np

class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, config):
        self.epsilon = config.get('epsilon')

    def choose_action(self, timestep, network, environment):
        if np.random.rand() < epsilon:
            # network의 Action argmax 반환
            pass
        else:
            # Environment의 Action Space 중 무작위 반환
            pass
            