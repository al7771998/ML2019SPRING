import sys
import numpy as np

f_test = sys.argv[1]
f_out = sys.argv[2]

w = np.load('strong_model.npy')
f_test = open(f_test,'r')
f_out = open(f_out,'w')
ith_row = 0
test_x = []
f_out.write('id,value\n')
for element in f_test.readlines():
	element = element.split()
	element = element[0].split(',')
	for i in range(9): #element[2:11]: 9 to 5
		if element[i+2] == 'NR': ## all below 2 change to 6
			test_x.append(0.0)
		elif float(element[i+2])< 0.0:
			if i == 0:
				if element[i+1+2] == 'NR':
					tmp = 0.0
				else:
					tmp = float(element[i+1+2])
			elif i == 8:
				if element[i-1+2] == 'NR':
					tmp = 0.0
				else:
					tmp = float(element[i-1+2])
			else:
				if element[i+1+2] == 'NR':
					tmp = 0.0
				else:
					tmp = float(element[i+1+2])
				if element[i-1+2] == 'NR':
					tmp += 0.0
				else:
					tmp += float(element[i-1+2])
				tmp /= 2.0
			test_x.append(tmp)
		else:
			test_x.append(float(element[i+2])) ## until here 2 change to 6
	ith_row += 1 
	if ith_row % 18 == 0:
		test_x.append(1.0)
		test_x = np.array(test_x)
		test_y = np.dot(w.transpose(),test_x)
		f_out.write(element[0] + ',' + str(int(round(test_y[0]))) + '\n')
		test_x = []
f_out.close()
f_test.close()