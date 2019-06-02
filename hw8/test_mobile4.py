from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,GaussianNoise,\
Flatten,Dropout,BatchNormalization,DepthwiseConv2D,GlobalAveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from keras.callbacks import CSVLogger
from keras.optimizers  import Adam
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
import random
import sys
import os

np.random.seed(6412)
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
#test_x /= 255
f.close()
#--------動以下就好--------
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,\
height_shift_range=0.2,shear_range=0.1,zoom_range=[0.8,1.2],\
fill_mode='constant', horizontal_flip=True)
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
#-------depthwise conv 2d layers 1-------
model.add(DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(Conv2D(filters=24, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.1))
#-------depthwise conv 2d layers 2-------
model.add(DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False))
model.add(BatchNormalization(axis = -1))
model.add(LeakyReLU())
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.1))
#-------depthwise conv 2d layers 3-------
model.add(DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False))
model.add(BatchNormalization(axis = -1))
model.add(LeakyReLU())
model.add(Conv2D(filters=48, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))
#-------depthwise conv 2d layers 4-------
model.add(DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False))
model.add(BatchNormalization(axis = -1))
model.add(LeakyReLU())
model.add(Conv2D(filters=84, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.3))

#Fully connected
model.add(Flatten())
model.add(Dense(7, activation='softmax')) 
print(model.summary())
model_weight = np.load('model_weights_4.npz', allow_pickle=True)['weights']
model_weight.tolist()
model.set_weights(model_weight)
test_y = model.predict_classes(test_x)  
print(test_y)
f = open(sys.argv[2],'w')
f.write('id,label\n')
for i in range(7178):
	f.write(str(i) + ',' + str(test_y[i]) + '\n')
f.close()