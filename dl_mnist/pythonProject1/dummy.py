import os
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
k  = torch.tensor([[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]])
k = k.view(1,-1)
print(k)