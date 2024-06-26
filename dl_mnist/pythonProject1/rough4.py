import matplotlib.pyplot as plt
import numpy as np
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

transform = transforms.Compose([transforms.Resize((200,200)),transforms.ToTensor()])

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
train_loader = DataLoader(custom_dataset,batch_size=1,shuffle=False)
'''
dataiter = iter(train_loader)
images = next(dataiter)
print(images[0].shape)
z = images[0][0]*255
x = z.permute(1, 2, 0)
j = x.numpy()
plt.imshow(j)
plt.show() '''
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self._body = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 3, kernel_size=(50, 50), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False)
        )
        self._head = nn.Sequential(
            nn.Linear(3, 3)
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
def validate(pathh):
    running_correct = 0
    transform_v = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])
    custom_dataset_v = CustomDataset(pathh, transform_v)
    test_loader = DataLoader(custom_dataset_v, batch_size=8, shuffle=False)
    model.eval()
    runacc = 0
    with torch.no_grad():
        for (x_val, y_val) in test_loader:
            y = model(x_val)
            prob = nn.functional.softmax(y, dim=1)
            y_pred = prob.argmax(dim=1)
            correct = torch.sum(y_pred == y_val)
            running_correct += correct
    return running_correct.item() / len(test_loader.dataset)
epochs = 20
model = MLP()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
for i in range(epochs):
    trainloss,trainacc = train()
    print(f'epoch: {i} , epoch_loss = {trainloss}, epoch_acc = {trainacc}')
acc = validate(testrootdir)
print(acc)




