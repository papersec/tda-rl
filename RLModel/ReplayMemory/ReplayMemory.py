from collections import deque
import numpy as np

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def put(self, transition):
        if self.memory.count() == self.memory.maxlen:
            self.memory.popleft()

        self.memory.append(transition)
    
    def sample(self, size):
        indexes = np.random.randint(len(self.memory), size=size)

        return tuple(self.memory[i] for i in indexes)