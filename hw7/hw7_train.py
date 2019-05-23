import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import load_model
from skimage.io import imread 
from keras.callbacks import CSVLogger,ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
X_train = np.load('X_train.npy')
X_train /= 255.

input_img = Input(shape=(32, 32, 3))

x = Conv2D(120, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(80, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(50, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(50, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(80, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(120, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
callbacks = []
modelcheck = ModelCheckpoint('encoder_4.h5',monitor='acc',verbose=1, save_best_only=True, mode='max', period=1)
callbacks.append(modelcheck)
train_history = autoencoder.fit(X_train,X_train,batch_size=200,epochs=20,\
                                verbose=1,shuffle=False,callbacks=callbacks)