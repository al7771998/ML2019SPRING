from keras import backend as K
from keras.models import load_model
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import random
import sys
import tensorflow as tf
import matplotlib
from skimage.util import random_noise
from skimage.transform import resize
import skimage.segmentation as seg
from skimage.color import gray2rgb, rgb2gray
from lime import lime_image
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
random.seed(1025)
y_label = np.zeros(7)
f = open(sys.argv[1],'r')
f.readline()
times, x, y = 0, [], []
label_first = np.zeros(7)
while times < 28709:
	seq = f.readline().split(',')
	y.append(int(seq[0]))
	x.append([float(i) for i in seq[1].split()])
	if y_label[int(seq[0])] == 0:
		label_first[int(seq[0])] = times
	y_label[int(seq[0])] += 1
	times += 1
f.close()
train_x = np.array(x).reshape(28709,48,48,1)
mean_x = np.sum(train_x, axis = 0) / np.size(train_x,0)
var_x = np.var(train_x, axis = 0)
var_x[var_x == 0] = 1e-10
train_x = (train_x - mean_x) / np.sqrt(var_x)
train_y = np.array(y)
model = load_model('model_final_3.h5')
print(model.summary())
image_num = 0#random.randint(0,28709)
#Saliency Map

input_tensor = [model.input]

for i in range(7):
	gradients = model.optimizer.get_gradients(model.output[0][i],model.input)
	sal = K.function(input_tensor, gradients)
	x_value = np.expand_dims(train_x[int(label_first[i])], axis=0)
	mask = sal([x_value])[0][0]
	plt.figure(1)
	plt.imshow(mask.reshape((48,48)),interpolation="nearest",cmap="jet")
	plt.colorbar()
	plt.savefig(sys.argv[2]+'fig1_%d.jpg'%(i))
	plt.close()

#Filter visualization
filt_im = np.zeros((1,48,48,1))
step,iter_n = 0.2,100
#filters = model.layers[15].get_weights()[0][:,:,0,:]
#plt.title('Filters of layer %s'%(model.layers[4].name))
get_layer_output = K.function([model.input],[model.layers[14].output])
np.random.seed(1025)
layer_output = get_layer_output([train_x[image_num].reshape(1,48,48,1)])[0]
for i in range(32):
	#print(i)
	if K.image_data_format() == 'channels_first':
		loss = K.mean(model.layers[14].output[:, i, :, :])
	else:
		loss = K.mean(model.layers[14].output[:, :, :, i])
	grads = K.gradients(loss, model.input)[0]
	iterate = K.function([model.input], [loss, grads])
	if K.image_data_format() == 'channels_first':
		gauss = np.random.random((1, 1, 48, 48))
	else:
		gauss = np.random.random((1, 48, 48, 1))
	for j in range(iter_n):
		loss_value, grads_value = iterate([gauss])
		gauss += grads_value * step
	plt.figure(1)
	plt.subplot(4,8,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(gauss.reshape((48,48)),interpolation="nearest",cmap="pink")
	if K.image_data_format() == 'channels_first':
		filt_im = np.array(layer_output[0,i,:,:])
	else:
		filt_im = np.array(layer_output[0,:,:,i])
	plt.figure(2)
	plt.subplot(4,8,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(filt_im,interpolation="nearest",cmap="pink")
plt.figure(1)
plt.suptitle('Filters of layer %s'%(model.layers[14].name))
plt.savefig(sys.argv[2]+'fig2_1.jpg')
plt.close()
plt.figure(2)
plt.suptitle('Output of layer %s (Given image %d)'%(model.layers[14].name, image_num))
plt.savefig(sys.argv[2]+'fig2_2.jpg')
plt.close()
#Lime

predict_fn_cnn = lambda x: model.predict_proba(np.expand_dims(rgb2gray(x),axis = 3)).astype(float)

for i in range(7):
	explainer = lime_image.LimeImageExplainer()
	np.random.seed(1025)
	explaination = explainer.explain_instance(image=train_x[int(label_first[i])].reshape((48,48)),\
	classifier_fn=predict_fn_cnn,segmentation_fn=seg.slic)
	image, mask = explaination.get_image_and_mask(label=i,\
	positive_only=False,hide_rest=False,num_features=5,min_weight=0.0)
	image = (image * np.sqrt(var_x) + mean_x)/ 255.0
	image = np.clip(image, 0., 1.)
	plt.imsave(sys.argv[2]+'fig3_%d.jpg'%(i),image)

#grad-cam
loss = model.output[0,train_y[image_num]]
conv_output =  model.layers[4].output
grads = K.gradients(loss, conv_output)[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-7) 
gradient_function = K.function([model.input], [conv_output, grads])
output, grads_val = gradient_function([train_x[image_num].reshape((1,48,48,1))])
output, grads_val = output[0, :], grads_val[0, :, :, :]
weights = np.mean(grads_val, axis = (0, 1))
cam = np.ones(output.shape[0 : 2], dtype = np.float32)
for i, w in enumerate(weights):
	cam += w * output[:, :, i]
cam = resize(cam,(48,48))
cam = np.maximum(cam, 0)
image = train_x[image_num].reshape((48,48))
image -= np.min(image)
image = np.minimum(image, 1.0)
plt.figure(1)
plt.imshow(image)
plt.imshow(cam, cmap = 'jet', alpha = 0.5)
plt.savefig(sys.argv[2]+'fig4_0.jpg')





