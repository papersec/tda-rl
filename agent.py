from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from network import QNetwork

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 1024
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4

class Agent:
    def __init__(self):
        self.q_local = QNetwork().cuda()
        self.q_target = QNetwork().cuda()
        self.optimizer = optim.RMSprop(self.q_local.parameters(), lr=0.00005, momentum=0.95)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def act(self, state, eps=0.0):
        state = state.unsqueeze(0).cuda()
        self.q_local.eval()
        with torch.no_grad():
            q_values = self.q_local(state)
        self.q_local.train()

        if np.random.uniform() < eps:
            return np.random.choice(4)
        else:
            return np.argmax(q_values.cpu().detach().numpy())
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.q_target(next_states.cuda()).cpu().detach().max(1).values.unsqueeze(1)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        Q_expected = self.q_local(states.cuda()).gather(dim=1, index=actions.unsqueeze(1).cuda())

        loss = F.mse_loss(Q_expected, Q_targets.cuda())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(TAU)
    
    def soft_update(self, tau):
        for target_param, local_param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def append(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences_idx = np.random.choice(len(self), size=self.batch_size)
        # print("experiences_idx", experiences_idx)
        experiences = [self.memory[i] for i in experiences_idx]

        states = torch.cat([e.state.unsqueeze(0) for e in experiences]).float()
        actions = torch.cat([torch.Tensor([e.action]) for e in experiences]).long()
        rewards = torch.cat([torch.Tensor([e.reward]) for e in experiences]).float()
        next_states = torch.cat([e.next_state.unsqueeze(0) for e in experiences]).float()
        dones = torch.cat([torch.Tensor([e.done]) for e in experiences]).float()

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)