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
from keras.callbacks import CSVLogger,ModelCheckpoint
import numpy as np
import random
import sys
import os

np.random.seed(6412)
f = open(sys.argv[1],'r')
f.readline()
times, x, y = 0, [], []
while times < 28709:
	seq = f.readline().split(',')
	y.append(int(seq[0]))
	x.append([float(i) for i in seq[1].split()])
	times += 1
train_x = np.array(x).reshape(28709,48,48,1)
mean_x = np.sum(train_x, axis = 0) / np.size(train_x,0)
var_x = np.var(train_x, axis = 0)
var_x[var_x == 0] = 1e-10
train_x = (train_x - mean_x) / np.sqrt(var_x)
np.savez_compressed('mean_n_var',mean=mean_x,var=var_x)
#train_x /= 255
val_x = train_x[27564:,:]
train_x = train_x[:27564,:]
train_y = np.array(y)
val_y = train_y[27564:]
train_y = train_y[:27564]
val_y = np_utils.to_categorical(val_y)
train_y = np_utils.to_categorical(train_y)
f.close()
#--------動以下就好--------
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,\
height_shift_range=0.2,shear_range=0.1,zoom_range=[0.8,1.2],\
fill_mode='constant', horizontal_flip=True)
datagen.fit(train_x)
'''model = MobileNet()
model.save_weights('model_mobile.h5')'''
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
'''model.add(Dense(128, activation='relu'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(Dropout(0.5))'''
model.add(Dense(7, activation='softmax')) 
print(model.summary())
optim = Adam(lr = 0.002)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy']) 
callbacks = []
csvLogger = CSVLogger("log_nc.csv", separator=",", append=False)
callbacks.append(csvLogger)
modelcheck = ModelCheckpoint('model_mobile4.h5',monitor='val_acc',verbose=1,\
                             save_best_only=True, save_weights_only=True, mode='max', period=1)
callbacks.append(modelcheck)

train_history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=128),\
steps_per_epoch=5*len(train_x)/128, epochs=250, verbose=2, shuffle = True,\
validation_data = (val_x,val_y), callbacks = callbacks)  
model_weight = model.get_weights()
model_weight = np.asarray(model_weight)
for i in range(len(model_weight)):
    model_weight[i] = model_weight[i].astype(np.float16)
np.savez_compressed('model_weights',weights = model_weight)