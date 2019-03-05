import sys
import numpy as np

f_in = 'train.csv'
## Data preprocessing
f_in = open(f_in,'r',encoding = 'big5')
Data = []
f_in.readline()
ith_row = 0
for element in f_in.readlines():
	element = element.split()
	element = element[0].split(',')
	for i in range(24):
		if element[2] == 'PM2.5':
			if float(element[i+3])< 0.0:
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
				Data.append(tmp)
			else:
				Data.append(float(element[i+3]))
f_in.close()
Data_l = int(len(Data)/12)
train_x, train_y = [],[]
for month in range(12):
	for sample in range(month * Data_l + 9,(month + 1) * Data_l): ## 9 changes to 5
		new_dat = []
		for i in Data[sample - 9:sample]:
			new_dat.append(i)
		new_dat.append(1.0)
		train_x.append(new_dat)
		train_y.append(Data[sample])
len_y = len(train_y)
train_x = np.array(train_x)
train_y = np.array(train_y).reshape((len_y,1))

## Training
w,lr,times = np.ones((10,1)),40.0,200000 ## 10 change to 6
prev_gra = 0.0
lamb = 0.01
while times > 0:
	opt_y = np.dot(train_x,w)
	Loss = opt_y - train_y
	#print(Loss)
	gradient = 2 * np.dot(train_x.transpose(),Loss)
	gradient += 2 * lamb * w
	#print(gradient)
	prev_gra += gradient**2
	ada = np.sqrt(prev_gra)
	w -= lr * gradient / ada
	times -= 1

np.save('strong_model_9',w)
'''
w = np.load('model.npy')
f_test = open(f_test,'r')
f_out = open(f_out,'w')
ith_row = 0
test_x = []
f_out.write('id,value\n')
for element in f_test.readlines():
	element = element.split()
	element = element[0].split(',')
	for i in element[2:11]:
		if i == 'NR':
			test_x.append(0.0)
		else:
			test_x.append(float(i))
	ith_row += 1 
	if ith_row % 18 == 0:
		test_x.append(1.0)
		test_x = np.array(test_x)
		test_y = np.dot(w.transpose(),test_x)
		f_out.write(element[0] + ',' + str(int(round(test_y[0]))) + '\n')
		test_x = []
f_out.close()
f_test.close()
'''


