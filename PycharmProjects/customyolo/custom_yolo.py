import os
import glob as glob
import matplotlib.pyplot as plt
import cv2
#import requests
import random
import numpy as np
np.random.seed(42)

import os
import zipfile

if not os.path.exists('drone.zip'):
    # Download the file
    os.system('curl -L "https://www.dropbox.com/sh/he3b4skcbvp1625/AAD0FJiGmBkQfSPtvG4yTF81a?dl=1" -o drone.zip')

# Unzip the file
with zipfile.ZipFile('drone.zip', 'r') as zip_ref:
    zip_ref.extractall('drone_dataset')

# Verify the extraction
if os.path.exists('drone_dataset'):
    print("Files have been successfully downloaded and extracted.")
else:
    print("There was an error downloading or extracting the files.")
