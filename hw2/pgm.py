import sys
import numpy as np
import pandas as pd
import math
import random
Data = pd.read_csv(sys.argv[3])

train_x = Data.values
train_x = np.delete(train_x,[31,32,33,34,35,36,37],axis = 1)
#train_x = np.c_[train_x,np.array(salary).reshape(32561,1)]
Data = pd.read_csv(sys.argv[4])
train_y = Data.values

#for i in range(np.size(train_x,0)):
#	train_x[i][1] /= 200000
'''for i in range(np.size(train_x,0)):
	train_x[i] = train_x[i] / np.linalg.norm(train_x[i])'''
num1, num2 = 0, 0 ##num1 is 1 and num2 is 0
for i in range(np.size(train_y)):
	if train_y[i] == 1:
		num1 += 1
	else:
		num2 += 1
mu1 = np.zeros((1,np.size(train_x,1)))
mu2 = np.zeros((1,np.size(train_x,1)))
Pc1, Pc2 = num1/(num1+num2), num2/(num1+num2)
for i in range(np.size(train_y)):
	if train_y[i] == 1:
		mu1 += train_x[i]
	else:
		mu2 += train_x[i]
mu1 /= num1
mu2 /= num2
#print(mu1,mu2)
sigma1 = np.zeros((np.size(train_x,1),np.size(train_x,1)))	
sigma2 = np.zeros((np.size(train_x,1),np.size(train_x,1)))		
for i in range(np.size(train_y)):
	if train_y[i] == 1:
		sigma1 += np.dot((train_x[i] - mu1).T, train_x[i] - mu1)
	else:
		sigma2 += np.dot((train_x[i] - mu2).T, train_x[i] - mu2)
sigma = sigma1 * Pc1 / num1 + sigma2 * Pc2 / num2
sigma_inv = np.linalg.pinv(sigma)
z = np.dot(train_x,np.dot(mu1-mu2,sigma_inv).T) - np.dot(np.dot(mu1,sigma_inv),mu1.T)/2
z += (np.dot(np.dot(mu2,sigma_inv),mu2.T)/2 + math.log(num1/num2))
opt_y = np.around(1 / (1 + np.exp(-z)))
print(np.sum(np.abs(opt_y-train_y))/np.size(train_y))
Data = pd.read_csv(sys.argv[5])
'''salary = []
for j in range(16281):
	for i in range(len(Nation_list)):
		if Data.iat[j,i+64] == 1:
			salary.append(GNI[i])'''

test_x = Data.values
test_x = np.delete(test_x,[31,32,33,34,35,36,37],axis = 1)
#test_x = np.c_[test_x,np.array(salary).reshape(16281,1)]
#for i in range(np.size(test_x,0)):
#	test_x[i][1] /= 200000
'''for i in range(np.size(test_x,0)):
	test_x[i] = test_x[i] / np.linalg.norm(test_x[i])'''
x_size = np.size(test_x,0)
test_y = np.zeros((x_size,1))
z = np.dot(test_x,np.dot(mu1-mu2,sigma_inv).T) - np.dot(np.dot(mu1,sigma_inv),mu1.T)/2
z += (np.dot(np.dot(mu2,sigma_inv),mu2.T)/2 + math.log(num1/num2))
test_y = 1 / (1 + np.exp(-z))
id_list = np.arange(1,int(np.size(test_y)+1)).reshape((np.size(test_y),1)).astype(int)
test_y = np.concatenate((id_list,np.around(test_y).astype(int)), axis = 1)
df = pd.DataFrame(test_y, columns=["id","label"])
df.to_csv(sys.argv[6],index=False)









