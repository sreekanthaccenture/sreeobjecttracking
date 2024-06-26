import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.datasets import ImageFolder
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import time
import torch.utils.data as Data
from torch import Tensor

Training_folder= r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Training_data"
ImageFolder(Training_folder)
IMG_WIDTH=200
IMG_HEIGHT=200
Train_folder= r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Training_data"
Test_folder= r"C:\Users\sreekanth.maramreddy\Desktop\New folder\data\Testing_Data"
def create_dataset(Train_folder):
    img_data_array=[]
    class_name=[]
    classes = {'driving_license': [1,0,0], 'others': [0,1,0], 'social_security': [0,0,1]}
    for PATH in os.listdir(Train_folder):
        for file in os.listdir(os.path.join(Train_folder, PATH)):
            image_path= os.path.join(Train_folder, PATH,  file)
            image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float64')
            # image /= 255
            if len(image.shape) == 3:
                img_data_array.append(np.array(image).reshape([3, 200, 200]))
                class_name.append(classes[PATH])
    return img_data_array, class_name
Train_img_data, train_class_name = create_dataset(Train_folder)
Test_img_data, test_class_name =create_dataset(Test_folder)
torch_dataset_train = Data.TensorDataset(Tensor(np.array(Train_img_data)), Tensor(np.array(train_class_name)))
torch_dataset_test = Data.TensorDataset(Tensor(np.array(Test_img_data)), Tensor(np.array(test_class_name)))
trainloader = torch.utils.data.DataLoader(torch_dataset_train, batch_size=1, shuffle=False)
testloader = torch.utils.data.DataLoader(torch_dataset_test, batch_size=1, shuffle=False)
dataiter = iter(trainloader)
images = next(dataiter)
print(images[3].shape)
#z = images[0][0]*255
#x = z.permute(1, 2, 0)
#j = x.numpy()
#plt.imshow(j)
#plt.show()
'''class MLP(nn.Module):
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

model = MLP()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)

def train():
    model.train()
    runloss = 0
    runacc = 0
    for x_train,y_train in trainloader:
        #x_train = x_train.view(x_train.shape[0],-1)

        optimizer.zero_grad()
        y = model(x_train)
        print(y)
        print(y_train)
        losss = loss_function(y,y_train)
        runloss = runloss + losss.item()

        losss.backward()
        optimizer.step()
        break
    return runloss/len(trainloader)

for i in range(20):
    trainloss = train()
    print(f'epoch: {i} , epoch_loss = {trainloss}')
    break '''
