import numpy as np
import torch
scores = np.array([0.9, 0.85, 0.8, 0.95, 0.9, 0.7])
indices = np.argsort(scores)[::-1]
print(indices)