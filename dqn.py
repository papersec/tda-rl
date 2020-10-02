import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

N_EPISODES = 1000
MAX_T = 50000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.997

from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(84,110)),
    transforms.CenterCrop(84),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

from collections import deque

class ObsBuffer:
    def __init__(self, tau=4):
        self.tau = tau
        self.buffer = deque([torch.zeros(size=(210,160,3), dtype=torch.float32) for _ in range(tau + 1)])
    
    def append(self, x):
        self.buffer.append(x)

    def observe(self):
        observation = [np.maximum(self.buffer[i-1], self.buffer[i]) for i in range(1, self.tau+1)]
        observation = [preprocess(img) for img in observation]
        return torch.cat(observation)


if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    from agent import Agent
    agent = Agent()
    scores = []

    eps = EPS_START
    for i_episode in range(N_EPISODES):
        score = 0
        obs_buffer = ObsBuffer()
        obs_buffer.append(env.reset())
        state = obs_buffer.observe()
        for t in range(MAX_T):
            action = agent.act(state, eps) # NEED TO IMPLEMENT
            # print("Action:", action)
            next_screen, reward, done, _ = env.step(action)
            obs_buffer.append(next_screen)
            next_state = obs_buffer.observe()

            agent.step(state, action, reward, next_state, float(done)) # NEED TO IMPLEMENT
            state = next_state
            score += reward

            if done:
                break
        scores.append(score)
        eps = max(EPS_END, EPS_DECAY * eps)
        if i_episode % 10 == 9:
            plt.plot(scores)
            plt.savefig("result/score"+str(i_episode+1)+".png")
            print("Episode {}\tscore: {:.2f}".format(i_episode, scores[i_episode]))