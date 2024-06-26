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

transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Lambda(lambda x : x/255.0)])

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
        self.model = nn.Sequential(
            nn.Linear(3*28*28,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
    def forward(self,inp):
        return  self.model(inp)

def train():
    model.train()
    runloss = 0
    runacc = 0
    for x_train,y_train in train_loader:
        x_train = x_train.view(x_train.shape[0],-1)
        y = model(x_train)
        losss = loss_function(y,y_train)
        runloss = runloss + losss.item()
        y_pred = y.argmax(dim=1)
        correct = torch.sum(y_pred == y_train)
        runacc = runacc + correct
        optimizer.zero_grad()
        losss.backward()
        optimizer.step()
    return runloss/len(train_loader) , runacc.item()/len(train_loader.dataset)
def validate():
    pass
epochs = 10
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

img = r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Testing_Data\social_security\53.jpg"
p = cv2.imread(img,cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(rgb_image, (28, 28))
l = torch.tensor(resized_image)
l = l/255.0
l = l.view(1,-1)
z = model(l)
prob = nn.functional.softmax(z)
n = prob.argmax(dim=1)
print(n)



