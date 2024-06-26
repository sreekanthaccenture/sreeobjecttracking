import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import os
torch.manual_seed(0)

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transforms.ToTensor())
validation_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

class mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)

def train():
    model.train()
    run_loss = 0
    run_correct = 0
    for (x_train, y_train) in train_loader:
        x_train = x_train.view(x_train.shape[0], -1).to(device)
        y = model(x_train)

        loss = loss_function(y, y_train.to(device))
        run_loss += loss.item()

        y_pred = y.argmax(dim=1)
        correct = torch.sum(y_pred == y_train.to(device))
        run_correct += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return run_loss / len(train_loader), run_correct.item() / len(train_loader.dataset)

def val():
    model.eval()
    running_loss = 0
    running_correct = 0
    with torch.no_grad():
        for (x_val, y_val) in validation_loader:
            x_val = x_val.view(x_val.shape[0], -1).to(device)
            y = model(x_val)
            prob = nn.functional.softmax(y, dim=1)
            y_pred = prob.argmax(dim=1)
            correct = torch.sum(y_pred == y_val.to(device))
            running_correct += correct

            loss = loss_function(y, y_val.to(device))
            running_loss += loss.item()
    return running_loss / len(validation_loader), running_correct.item() / len(validation_loader.dataset)

epochs = 20
model = mlp().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

start_time = time.time()
for ep in range(epochs):
    train_loss, train_acc = train()
    val_loss, val_acc = val()
    print(f'epoch = {ep}, train loss: {train_loss}, train accuracy: {train_acc}, val loss: {val_loss}, val accuracy: {val_acc}')
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

end_time = time.time()
total_time = end_time - start_time
print(f'Total time taken for training: {total_time} seconds')

models = 'models'
if not os.path.exists(models):
    os.makedirs(models)
model_file_name = 'mist_cpu.pt'
model_path = os.path.join(models,model_file_name)
model.to('cpu')
torch.save(model.state_dict(),model_path)
