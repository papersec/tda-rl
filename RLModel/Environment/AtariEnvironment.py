import gym

class AtariEnvironment(Environment):
    def __init__(self, env_name):
        self.env = gym.make(env_name).unwrapped

    def get_observation(self):
        return self.env.
    
    def conduct_action(self, action):
        observation, reward, done, info =  self.env.step(action)
        return observation, reward, done
    
