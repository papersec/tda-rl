class ObservationPreprocessor(object):
    def __init__(self):
        pass

    def put(self, image):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

