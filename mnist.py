# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
'''
MNIST data setup
'''

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)


# %%
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


# %%
from matplotlib import pyplot
import numpy as np


# %%
'''numpy array to torch tensor'''
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)


# %%
'''GPU Using'''
print(torch.cuda.is_available())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# %%
import torch.nn.functional as F

loss_func = F.cross_entropy


# %%
import torch.nn as nn

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features= nn.Sequential(
            # 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 3
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),
            # 5
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 6
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 7
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),
            # 8
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 9
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 10
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 11
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 12
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),

            # 13
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=7),
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.features(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x
    


# %%
model = Mnist_CNN()
model.to(device)

from torchsummary import summary
print(summary(model, (1,28,28)))


# %%
from torch import optim
lr = 0.0025  # learning rate
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)


# %%
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

bs = 16


train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


# %%
epochs = 10000  # how many epochs to train for

g_losses, g_accuracies, g_valid_losses, g_valid_accuracies = [], [], [], []

def fit():
    for epoch in range(epochs):
        model.train()
        losses = []
        accuracies = []
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_func(pred, yb)

            with torch.no_grad():
                losses.append(loss.cpu().item())
                accuracies.append((torch.argmax(pred, dim=1) == yb).cpu().float().mean().item())

            loss.backward()
            opt.step()
            opt.zero_grad()

        '''Validation'''
        model.eval()
        with torch.no_grad():
            valid_losses = []
            valid_accuracies = []
            for xb, yb in valid_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                valid_losses.append(loss_func(pred, yb).cpu().item())
                valid_accuracies.append((torch.argmax(pred, dim=1) == yb).cpu().float().mean().item())
        print(epoch, np.average(losses), "\t", np.average(accuracies), "\t", np.average(valid_losses), "\t", np.average(valid_accuracies))
        g_losses.append(np.average(losses))
        g_accuracies.append(np.average(accuracies))
        g_valid_losses.append(np.average(valid_losses))
        g_valid_accuracies.append(np.average(valid_accuracies))
        
        continue # Skip Image Save

        epoch_step = (epochs // 20)
        if (epoch + 1) % epoch_step == 0:
            fig, ax1 = pyplot.subplots()

            ax1.set_xlabel('epoch')
            ax1.set_ylabel('loss')
            ax1.plot(g_losses[epoch+1-epoch_step:], label="train loss", color='tab:orange')
            ax1.plot(g_valid_losses[epoch+1-epoch_step:], label="valid loss", color='tab:red')

            ax2 = ax1.twinx()
            ax2.set_ylabel('accuracy')
            ax2.plot(g_accuracies[epoch+1-epoch_step:], label="train accuracy", color='tab:purple')
            ax2.plot(g_valid_accuracies[epoch+1-epoch_step:], label="valid accuracy", color='tab:blue')

            pyplot.savefig("hist_plot/"+str(epoch+1)+".png")

fit()


# %%
torch.save(model.state_dict(), "trained_model/pnet200905.pth")