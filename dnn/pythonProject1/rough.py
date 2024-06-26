import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

'''k = np.array([[[[-1.51507401e+00]],[[-8.91683221e-01]],[[-8.27306330e-01]],[[-1.10410154e+00]],[[-4.08073330e+00]]]])
final_outputs = k.reshape(5, 1)
print(final_outputs)
label_id = np.argmax(final_outputs)
print(label_id)
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
print(probs)
final_prob = np.max(probs) * 100.0
print(final_prob)'''
num_images = 6
num_columns = 3
#num_rows = (num_images // num_columns) + (num_images % num_columns > 0)
print(num_images % num_columns > 0)
