import sys
import numpy as np
import pandas as pd
import random as random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.utils.validation import column_or_1d
random.seed(6187)#1475 2263 3487 5628 6639
##Training Part
"""
Data = pd.read_csv(sys.argv[3])

train_x = Data.values

train_x = np.delete(train_x,[31,32,33,34,35,36,37],axis = 1)
s_features = np.size(train_x,1)
'''train_x = train_x.astype(float)
mean_i = np.sum(train_x, axis = 0) / np.size(train_x,0)
var_i = np.var(train_x, axis = 0)
for i in range(np.size(train_x,0)):
	train_x[i] -= mean_i
	train_x[i] /= np.sqrt(var_i)'''
Data = pd.read_csv(sys.argv[4])
train_y = Data.values
train_y = column_or_1d(train_y)
#for i in range(np.size(train_x,0)):
#	train_x[i][1] /= 200000.0
#train_x = np.c_[train_x,np.ones((np.size(train_x,0),1))]



##cross validation preprocess end
model1 = GradientBoostingClassifier(learning_rate=0.1, \
n_estimators=500, subsample=1.0, min_samples_split=2, min_samples_leaf=1, \
min_weight_fraction_leaf=0., max_depth=3, min_impurity_decrease=0., \
random_state=1279, validation_fraction=0.1, tol=1e-6
)

cv_num_choose = 2378
rand_list = random.sample(range(np.size(train_x,0)),k = cv_num_choose)
cv_x = np.zeros((np.size(train_x,0) - cv_num_choose ,s_features))
cv_x_t = np.zeros((cv_num_choose,s_features))
cv_y = np.zeros((np.size(train_x,0) - cv_num_choose,1))
cv_y_t = np.zeros((cv_num_choose,1))
tr_num,cv_num = 0,0
for i in range(np.size(train_x,0)):
	if i in rand_list:
		cv_x_t[cv_num] = train_x[i]
		cv_y_t[cv_num] = train_y[i]
		cv_num += 1
	else:
		cv_x[tr_num] = train_x[i]
		cv_y[tr_num] = train_y[i]
		tr_num += 1
cv_y = column_or_1d(cv_y)
cv_y_t = column_or_1d(cv_y_t)


model1.fit(cv_x,cv_y)
predict = model1.predict(cv_x_t)
print("Loss:",np.sum(np.abs(predict - cv_y_t))/ cv_x_t.shape[0])
##cross validation end


#model2.fit(notUS_x,notUS_y)
model1.fit(train_x,train_y)
predict1 = model1.predict(train_x)

#predict2 = model2.predict(notUS_x)
print("Loss:",np.sum(np.abs(predict1 - train_y))/ train_x.shape[0])

joblib.dump(model1, 'gb_model.pkl')


#w = model.coef_.reshape((99,1))
#np.save('skmodel',w)
#w = np.load('skmodel.npy')
"""
## Testing Part
Data = pd.read_csv(sys.argv[5])
#for i in marital_status:
#	Data = Data.drop(i,axis = 1)
'''unk = Data['?_workclass'].values
for i in range(np.size(Data['?_workclass'].values)):
	if unk[i] == 1:
		Data.iat[i,6+random.randint(0,7)] = 1
Data.drop('?_workclass',axis = 1)'''
test_x = Data.values
test_x = np.delete(test_x,[31,32,33,34,35,36,37],axis = 1)
'''test_x = test_x.astype(float)
mean_i = np.sum(test_x, axis = 0) / np.size(test_x,0)
var_i = np.var(test_x, axis = 0)
var_i[var_i == 0] = 1.0
for i in range(np.size(test_x,0)):
	test_x[i] -= mean_i
	test_x[i] /= np.sqrt(var_i)'''
#test_x = np.c_[test_x,np.ones((np.size(test_x,0),1))]


model1 = joblib.load('gb_model.pkl')

test_y = model1.predict(test_x)
#test_notUS_y = model2.predict(test_notUS_x)
#test_y = np.zeros((np.size(test_x,0),1))
#num_US, num_notUS = 0,0
#for i in range(np.size(test_x,0)):
#	if i in US_list:
#		test_y[i][0] = test_US_y[num_US]
#		num_US += 1
#	else:
#		test_y[i][0] = test_notUS_y[num_notUS]
#		num_notUS += 1
tmp = np.arange(1,int(np.size(test_y)+1)).reshape((np.size(test_y),1)).astype(int)
test_y = np.concatenate((tmp,np.around(test_y).reshape((np.size(test_y),1)).astype(int)), axis = 1)
df = pd.DataFrame(test_y, columns=["id","label"])
df.to_csv(sys.argv[6],index=False)





