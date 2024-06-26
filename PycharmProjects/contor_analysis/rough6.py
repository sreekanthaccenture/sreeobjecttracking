import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
validate_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self._body = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self._head = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self._body(x)
        x = x.view(x.size()[0], -1)
        x = self._head(x)
        return x


def train():
    model.train()
    running_loss = 0
    running_correct = 0
    for (x_train, y_train) in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)  # Move to GPU
        y = model(x_train)
        loss = loss_function(y, y_train)
        running_loss += loss.item()
        y_pred = y.argmax(dim=1)
        correct = torch.sum(y_pred == y_train)
        running_correct += correct
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss/len(train_loader), running_correct.item()/len(train_loader.dataset)

def val():
    model.eval()
    running_loss = 0
    running_correct = 0
    with torch.no_grad():
        for (x_val, y_val) in validation_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)  # Move to GPU
            y = model(x_val)
            prob = nn.functional.softmax(y, dim=1)
            y_pred = prob.argmax(dim=1)
            correct = torch.sum(y_pred == y_val)
            running_correct += correct
            loss = loss_function(y, y_val)
            running_loss += loss.item()
    return running_loss/len(validation_loader), running_correct.item()/len(validation_loader.dataset)

num_epochs = 20
model = CNN().to(device)  # Move the model to GPU
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

print("Starting Training...")
for ep in range(num_epochs):
    train_loss, train_acc = train()
    val_loss, val_acc = val()
    print("Epoch: {}, Train Loss = {:.3f}, Train Acc = {:.3f} , Val Loss = {:.3f}, Val Acc = {:.3f}".format(ep, train_loss, train_acc, val_loss, val_acc))
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
torch.save(model.state_dict(), 'cnn_model.pth')
torch.save(model, 'cnn_model_complete.pth')