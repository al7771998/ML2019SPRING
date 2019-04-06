from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,GaussianNoise,Flatten,Dropout,BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
from keras.optimizers  import Adam
from keras.models import load_model
import numpy as np
import random
import sys
import os

f = open(sys.argv[1],'r')
f.readline()
times, x = 0, []
while times < 7178:
	seq = f.readline().split(',')
	x.append([float(i) for i in seq[1].split()])
	times += 1
test_x = np.array(x).reshape(7178,48,48,1)
mean_x = np.sum(test_x, axis = 0) / np.size(test_x,0)
var_x = np.var(test_x, axis = 0)
var_x[var_x == 0] = 1e-10
test_x = (test_x - mean_x) / np.sqrt(var_x)
f.close()
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,\
height_shift_range=0.2,shear_range=0.1,zoom_range=[0.8,1.2],\
fill_mode='constant', horizontal_flip=True)
datagen.fit(test_x)
model = load_model('model_final_3.h5')
test_y = model.predict_classes(test_x)  
print(test_y)
f = open(sys.argv[2],'w')
f.write('id,label\n')
for i in range(7178):
	f.write(str(i) + ',' + str(test_y[i]) + '\n')
f.close()