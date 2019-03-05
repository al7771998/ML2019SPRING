import sys
import numpy as np
import random
f_in = 'train.csv'
## Data preprocessing
f_in = open(f_in,'r',encoding = 'big5')
Data = [[] for i in range(18)]
f_in.readline()
ith_row = 0
for element in f_in.readlines():
	element = element.split()
	element = element[0].split(',')
	for i in range(24): ##element[3:27]:
		if element[i+3] == 'NR':
			Data[ith_row % 18].append(0.0)
		elif float(element[i+3])< 0.0:
			if i == 0:
				if element[i+1+3] == 'NR':
					tmp = 0.0
				else:
					tmp = float(element[i+1+3])
			elif i == 23:
				if element[i-1+3] == 'NR':
					tmp = 0.0
				else:
					tmp = float(element[i-1+3])
			else:
				if element[i+1+3] == 'NR':
					tmp = 0.0
				else:
					tmp = float(element[i+1+3])
				if element[i-1+3] == 'NR':
					tmp += 0.0
				else:
					tmp += float(element[i-1+3])
				tmp /= 2.0
			Data[ith_row % 18].append(tmp)
		else:
			Data[ith_row % 18].append(float(element[i+3]))
	ith_row += 1 
f_in.close()
Data_l = len(Data[1])
train_x, train_y = [],[]
'''for sample in range(9,Data_l): ## 9 changes to 5
	new_dat = []
	for i in range(18):
		for j in Data[i][sample - 9:sample]: ## 9 changes to 5
			new_dat.append(j)
	new_dat.append(1.0)
	train_x.append(new_dat)
	train_y.append(Data[9][sample]) 
'''
random.seed()
list_rand = random.choices(range(Data_l - 9), k = 3000)
for sample in list_rand:
	new_dat = []
	for i in range(18):
		for j in Data[i][sample:sample + 9]:
			new_dat.append(j)
	new_dat.append(1.0)
	train_x.append(new_dat)
	train_y.append(Data[9][sample + 9])
len_y = len(train_y)
train_x = np.array(train_x)
train_y = np.array(train_y).reshape((len_y,1))

## Training(4,200000)
w,lr,times = np.ones((163,1)),40.0,200000 ##163 change to 91
prev_gra = 0.0
while times > 0:
	opt_y = np.dot(train_x,w)
	Loss = opt_y - train_y
	#print(Loss)
	gradient = 2 * np.dot(train_x.transpose(),Loss)
	#print(gradient)
	prev_gra += gradient**2
	ada = np.sqrt(prev_gra)
	w -= lr * gradient / ada
	times -= 1

#np.save('strong_model',w)
test_x, test_y = [],[]
test_rand = random.choices(range(Data_l - 9),k = 600)
test_num = 600
for sample in test_rand:
	new_dat = []
	for i in range(18):
		for j in Data[i][sample:sample + 9]:
			new_dat.append(j)
	new_dat.append(1.0)
	test_x.append(new_dat)
	test_y.append(Data[9][sample + 9])
len_y = len(test_y)
test_x = np.array(test_x)
test_y = np.array(test_y).reshape((len_y,1))
result_y = np.dot(test_x,w)
Loss = result_y - test_y

ave = 0.0
for i in range(np.size(Loss)):
	ave += abs(Loss[i][0])
ave /= test_num
print(ave)
