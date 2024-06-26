import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

train_dataset = datasets.MNIST('./data', train=True,download=False,transform=transforms.ToTensor())
print(train_dataset)
train_loader = DataLoader(dataset=train_dataset)
print(train_loader)#<torch.utils.data.dataloader.DataLoader object at 0x000001E171DCA1B0>
size = len(train_loader)
mean = 0
stds = 0
for (image, y_train) in train_loader: # image, y_train are tensors
    image_reshape = image.view(28, -1)
    print(y_train)
    #mean = mean + (image_reshape.mean())
    #stds = stds + (image_reshape.std())
    #print(image_reshape.shape)#torch.Size([1, 1, 28, 28])
    #plt.imshow(image_reshape)
    #plt.show()
    break
#print(mean/size)
#print(stds/size)