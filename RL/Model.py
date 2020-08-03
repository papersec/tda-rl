'''
Code from https://github.com/udacity/deep-reinforcement-learning/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features = 16)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features = 32)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=32 * 9 * 9, out_features=256, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features = 256)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=256, out_channels=18, bias=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x