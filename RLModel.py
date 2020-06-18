import gym


class RLModel:

    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.time = 0

    def reset_game(self):
        self.env.reset()
        self.time = 0

    def conduct_action(self, action):
        reward, obs, done = self.env.step(action)
        return reward, obs, done
