import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import time
import cv2
import os
trainingrootdir = r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Training_data"
testrootdir = r = r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Testing_Data"

transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Lambda(lambda x : x/255.0)])

class CustomDataset(Dataset):
    def __init__(self,rootdir,transform = None):
        self.rootdir = rootdir
        self.transform = transform
        self.classes = os.listdir(rootdir)
        self.images = []
        self.labels = []
        for labels,classs in enumerate(self.classes):
            class_path = os.path.join(rootdir,classs)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path,img)
                self.images.append(img_path)
                self.labels.append(labels)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        p = Image.open(image_path).convert('RGB')
        if self.transform:
            p = self.transform(p)
        return p , torch.tensor(label)
custom_dataset = CustomDataset(trainingrootdir,transform)
train_loader = DataLoader(custom_dataset,batch_size=32,shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self._body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self._head = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
        )

    def forward(self,inp):
        x = self._body(inp)
        x = x.view(x.size()[0], -1)
        x = self._head(x)
        return x
def train():
    model.train()
    runloss = 0
    runacc = 0
    for x_train,y_train in train_loader:
        #x_train = x_train.view(x_train.shape[0],-1)
        optimizer.zero_grad()
        y = model(x_train)
        losss = loss_function(y,y_train)
        runloss = runloss + losss.item()
        y_pred = y.argmax(dim=1)
        correct = torch.sum(y_pred == y_train)
        runacc = runacc + correct
        losss.backward()
        optimizer.step()
    return runloss/len(train_loader) , runacc.item()/len(train_loader.dataset)
def validate():
    pass
epochs = 50
model = MLP()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
train_loss = []
train_acc = []
val_loss = []
val_acc = []
for i in range(epochs):
    trainloss,trainacc = train()
    print(f'epoch: {i} , epoch_loss = {trainloss}, epoch_acc = {trainacc}')




