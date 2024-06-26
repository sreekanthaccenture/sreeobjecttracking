import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
torch.manual_seed(0)
#size = (3,5,5)
#k = np.ones(size, dtype= int)
#size = (3,2,2)
#k = torch.ones(6, dtype= float)
#k = torch.ones(size, dtype= float)
#print(k)
#print(k.mean())
'''p = np.array([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]])
k = torch.tensor(p,dtype=float)
print(torch.mean(k))
print(torch.mean(k,dim=[1,2]))'''
'''k  = np.array([[10,1,3],[15,6,2],[8,1,3]])
print(k)
print(k.argmax(0))
print(k.argmax(1))'''
import os
#cwd = os.getcwd()
#print("Current working directory:", cwd)
'''path = "/"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print(dir_list)'''
#print(os.path.exists('data'))

'''z  = torch.tensor([[2.9045e-06, 1.4930e-07, 8.9732e-05, 1.1227e-03, 2.8357e-08, 3.5633e-06,
         8.7808e-12, 9.9874e-01, 8.9586e-06, 3.5630e-05]])
print(z.data)
print(z.data.max(dim = 1))
print(z.data.max(dim = 1)[0])
print(z.data.max(dim = 1)[1])'''
'''best_loss = torch.tensor(np.inf)
print(best_loss)'''
t = torch.tensor([[ -46.0687,  -33.2159,  -88.2575],
        [ -44.2938,  -36.1662,  -96.5480],
        [ -58.4186,  -45.1075, -117.6022],
        [ -62.9610,  -44.2847, -119.0229],
        [ -55.1724,  -41.3692, -110.6765],
        [ -48.8219,  -42.5667, -108.5369],
        [ -57.7341,  -44.5770, -115.5984],
        [ -48.6123,  -36.0683,  -98.4219]])
a = torch.tensor([[1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 0., 0.]])
y_pred = t.argmax(dim=1)
y_true = a.argmax(dim=1)
print(y_pred)
print(y_true)
print(torch.sum(y_pred == y_true))
