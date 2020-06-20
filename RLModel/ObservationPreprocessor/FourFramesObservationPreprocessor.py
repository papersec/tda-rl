from collections import deque

import ObservationPreprocessor

class FourFramesObservationPreprocessor(ObservationPreprocessor):
    def __init__(self, config):
        self.memory = deque(maxlen=5)
        # ~ config로부터 프레임 크기, 타입 정보 받아와 저장

    def put(self, image):
        if len(self.memory) == self.memory.maxlen:
            self.memory.popleft()

        self.memory.append(image)
    
    def get_observation(self):
        if not len(memory) == memory.maxlen:
            raise ValueError

        # ~ 메모리에서 프레임 가져오고 빈 부분은 Pool해 반환

