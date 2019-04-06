from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,GaussianNoise,Flatten,Dropout,BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from keras.callbacks import CSVLogger
from keras.optimizers  import Adam
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
model = Sequential()
#Layer 0
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(Dropout(0.1))
#Layer 1
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(Dropout(0.1))
#Layer 1.5
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.1))
#Layer 2
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))
#Layer 3
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))
#Layer 4
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.3))
#Layer 5

model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.3))

#Fully connected
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax')) 
print(model.summary())
optim = Adam(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy']) 
callbacks = []
csvLogger = CSVLogger("log2.csv", separator=",", append=True)
callbacks.append(csvLogger)
'''
train_history = model.fit(train_x,train_y, batch_size = 640, callbacks=callbacks,\
validation_data = (val_x,val_y), epochs=50, verbose=1, shuffle = True) 
'''

train_history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=128),\
steps_per_epoch=3*len(train_x)/128, epochs=50, verbose=1, shuffle = True,\
validation_data = (val_x,val_y), callbacks = callbacks)  
model.save('model1.h5')
train_history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=128),\
steps_per_epoch=3*len(train_x)/128, initial_epoch = 50, epochs=100, verbose=1, shuffle = True,\
validation_data = (val_x,val_y), callbacks = callbacks)   
model.save('model2.h5')
train_history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=128),\
steps_per_epoch=3*len(train_x)/128, initial_epoch = 100, epochs=150, verbose=1, shuffle = True,\
validation_data = (val_x,val_y), callbacks = callbacks)  
model.save('model_final_3.h5')






