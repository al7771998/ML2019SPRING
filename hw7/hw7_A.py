import os
import sys
import numpy as np 
from skimage.io import imread, imsave
from skimage import transform
import sys
IMAGE_PATH = sys.argv[1]
# Images for compression & reconstruction
test_image = [sys.argv[2]] 
# Number of principal components used
k = 5

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M
  
filelist = os.listdir(IMAGE_PATH) 
img_shape = (600,600,3)#imread(os.path.join(IMAGE_PATH,filelist[0])).shape 
img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    tmp = transform.resize(tmp,img_shape)
    img = tmp.flatten()
    img_data.append(img)

training_data = np.array(img_data).astype('float32')
# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 
# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(training_data.T, full_matrices = False)  
for x in test_image: 
    # Load image & Normalize
    picked_img = imread(os.path.join(IMAGE_PATH,x)) 
    picked_img = transform.resize(picked_img,img_shape)
    X = picked_img.flatten().astype('float32') 
    X -= mean #(67500,)
    # Compression
    weight = np.array([X.dot(u[:,i]) for i in range(k)])  #(1,5)
    # Reconstruction
    reconstruct = process(weight.dot(u[:,0:k].T) + mean)
    imsave(sys.argv[3], transform.resize(reconstruct.reshape(img_shape),(600,600,3))) 
average = process(mean)
imsave('average.jpg', average.reshape(img_shape))  
for x in range(5):
    eigenface = process(u[:,x])
    imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))  
for i in range(5):
    number = s[i] * 100 / sum(s)
    print(number)