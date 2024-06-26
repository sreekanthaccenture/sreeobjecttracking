import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import os
import matplotlib.pyplot as plt
torch.manual_seed(0)

test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class mlp(nn.Module):
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

model = mlp()
name = 'models'
model_file_name = 'mist_cpu.pt'
model_path = os.path.join(name,model_file_name)
print(model_path)
model.load_state_dict(torch.load(model_path))
def val(model,image):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        data = image.to(device)
        output = model(data)
        prob = nn.functional.softmax(output,dim = 1)
        print(prob)
        print(prob.data.max(dim = 1))
        pred_prob = prob.data.max(dim = 1)[0]
        pred_index = prob.data.max(dim = 1) [1]
        return pred_index.cpu()
for data,v in test_dataset:
    #data = data.view(28,28)
    #plt.imshow(data)
    #plt.show()
    data = data.view(data.shape[0], -1)
    print(data)
    k = val(model,data)
    print(v)
    print(k)
    print(type(k))
    break