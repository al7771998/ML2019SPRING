import numpy as np
import pandas as pd
from skimage.io import imread 
import sys
X_train = np.zeros((40000,32,32,3))
for i in range(40000):
    X_train[i] = imread(sys.argv[1]+'%06d.jpg'%(i+1))
    if i % 1000 == 0:
        print(i)
np.save('X_train',X_train)