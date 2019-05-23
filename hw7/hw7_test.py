import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import load_model
from skimage.io import imread 
from keras.callbacks import CSVLogger,ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
X_train = np.load('X_train.npy')
X_train /= 255.
np.random.seed(127)
'''input_img = Input(shape=(32, 32, 3))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callbacks = []
modelcheck = ModelCheckpoint('encoder.h5',monitor='val_loss',verbose=1, save_best_only=True, mode='min', period=1)
callbacks.append(modelcheck)
train_history = autoencoder.fit(X_train,X_train,validation_split=0.1,batch_size=256,epochs=100,\
                                verbose=1,shuffle=False,callbacks=callbacks)'''
autoencoder = load_model('encoder_4.h5')
#encoder_layer_name = 'max_pooling2d_21' #For 'encoder_0.h5'
encoder_layer_name = 'max_pooling2d_3' #For 'encoder_1.h5' and 'encoder_2.h5' and 'encoder_4.h5'
#encoder_layer_name = 'max_pooling2d_6' #For 'encoder_3.h5'
encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(encoder_layer_name).output)
X_encoded = encoder_model.predict(X_train)
X_encoded = X_encoded.reshape((40000,np.size(X_encoded,1) * np.size(X_encoded,2) * np.size(X_encoded,3)))
pca = PCA(n_components=720, whiten=True,random_state=127)
X_pca = pca.fit_transform(X_encoded)
kmeans = KMeans(n_clusters=2, random_state=127).fit(X_pca)
Y_train = kmeans.predict(X_pca)
#labels = np.array(labels)
Data_X = pd.read_csv(sys.argv[2])
X_test = Data_X.values
f = open(sys.argv[3],'w')
f.write('id,label\n')
for i in range(np.size(X_test,0)):
    num1,num2 = int(X_test[i][1])-1, int(X_test[i][2])-1
    if num1<1000 and num2<1000:
        print(num1+1,num2+1,Y_train[num1],Y_train[num2])
    if Y_train[num1]==Y_train[num2]:
        f.write(str(i) + ',1\n')
    else:
        f.write(str(i) + ',0\n')
f.close()