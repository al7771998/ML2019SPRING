import sys
import numpy as np
import pandas as pd
import random
random.seed(4096)
###Training Part Code:
'''
Data = pd.read_csv(sys.argv[3])
train_x = Data.values

Data = pd.read_csv(sys.argv[4])
train_y = Data.values
#for i in range(np.size(train_x,0)):
train_x = np.delete(train_x,[31,32,33,34,35,36,37],axis = 1)
train_x = train_x.astype(float)

mean_i = np.sum(train_x, axis = 0) / np.size(train_x,0)
var_i = np.var(train_x, axis = 0)
for i in range(np.size(train_x,0)):
	train_x[i] -= mean_i
	train_x[i] /= np.sqrt(var_i)
train_x = np.c_[train_x,np.ones((np.size(train_x,0),1))]

lamb_list = [1e-3, 1e-1, 1e1, 1e3]
#for lamb in lamb_list:
w,lr,times = np.ones((100,1))/1e2, 3e-6, 60000
while times > 0:
	opt_y = 1 / (1 + np.exp(-np.dot(train_x,w)))
	Loss = opt_y - train_y 
	gradient = 2 * np.dot(train_x.transpose(),Loss)
	opt_y = np.around(opt_y)
	if times % 1000 == 0:
		#print(gradient)
		print("Loss:",np.sum(np.abs(opt_y - train_y))/ train_x.shape[0])
	#gradient += 2 * lamb * w
	#prev_gra = gradient**2
	#ada = np.sqrt(prev_gra)
	w -= lr * gradient #/ ada
	times -= 1

np.save('model',w)
'''
## Testing Part
w = np.load('model.npy')
Data = pd.read_csv(sys.argv[5])
test_x = Data.values
test_x = np.delete(test_x,[31,32,33,34,35,36,37],axis = 1)
#for i in range(np.size(test_x,0)):
test_x = test_x.astype(float)
mean_i = np.sum(test_x, axis = 0) / np.size(test_x,0)
var_i = np.var(test_x, axis = 0)
var_i[var_i == 0.0] = 1.0
for i in range(np.size(test_x,0)):
	test_x[i] -= mean_i
	test_x[i] /= np.sqrt(var_i)
test_x = np.c_[test_x,np.ones((np.size(test_x,0),1))]

test_y = 1 / (1 + np.exp(-np.dot(test_x,w)))

tmp = np.arange(1,int(np.size(test_y)+1)).reshape((np.size(test_y),1)).astype(int)
test_y = np.concatenate((tmp,np.around(test_y).astype(int)), axis = 1)
df = pd.DataFrame(test_y, columns=["id","label"])
df.to_csv(sys.argv[6],index=False)


