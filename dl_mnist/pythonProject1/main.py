import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
torch.manual_seed(0)


train_dataset = datasets.MNIST('./data', train=True,download=False,transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data', train=False,transform=transforms.ToTensor())

batch_size = 32

train_loader = DataLoader(dataset=train_dataset,  batch_size=batch_size,shuffle=True)

validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size,shuffle=False)

class MLP(torch.nn.Module):
    def __init__(self):
        # Initialize super class
        super().__init__()

        # Build model using Sequential container
        self.model = nn.Sequential(
            # Add input layer
            nn.Linear(28*28, 512),
            # Add ReLU activation
            nn.ReLU(),
            # Add Another layer
            nn.Linear(512, 512),
            # Add ReLU activation
            nn.ReLU(),
            # Add Output layer
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # Forward pass
        return self.model(x)
def train():
    model.train()
    running_loss = 0
    running_correct = 0
    for (x_train, y_train) in train_loader:
        x_train = x_train.view(x_train.shape[0], -1)
        print(x_train)
        y = model(x_train)
        print(y)
        loss = loss_function(y, y_train)
        print(loss)
        print(loss.item())
        running_loss += loss.item()

        y_pred = y.argmax(dim=1)
        print(y_pred)
        correct = torch.sum(y_pred == y_train)
        print(correct)
        running_correct += correct
        print(running_correct)
        break

# Construct model
model = MLP()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
k = train()
