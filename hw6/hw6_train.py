#Word-Embedding RNN
from gensim.models import Word2Vec
import jieba
import pandas as pd
import numpy as np
import sys
import string
import pickle
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Adam
from keras.regularizers import l2
import emoji
import random
from keras.callbacks import CSVLogger,ModelCheckpoint
from keras.layers import TimeDistributed
from keras.utils import to_categorical
import ast
#Read data
random.seed(9487)
np.random.seed(9487)
Data_X = pd.read_csv(sys.argv[1])
array_X = Data_X.values
train_X = array_X[:,1]
jieba.load_userdict(sys.argv[4])
#print(list_rand[:20])
#random.shuffle(list_rand)
#print(list_rand[:20])
#np.random.shuffle(train_X)
#val_num = 107777
#val_X = train_X[val_num:119017]
Data_X = pd.read_csv(sys.argv[3])
array_X = Data_X.values
test_X = array_X[:,1]
seg_X = []
list_train = []
#list_val = []
list_test = []
list_punc = string.punctuation

#Preprocess data
for i in range(np.size(train_X)):
    '''seg = emoji.demojize(train_X[i])
    seg = jieba.cut(seg,cut_all=False)'''
    seg = jieba.cut(train_X[i],cut_all=False)
    seg = ' '.join(seg)
    seg = seg.split(' ')
    seg_pre = []
    for j in seg:
        if j != '' and j[0] not in list_punc and j[0] != '？' and j[0]!='～' and j[0]!='：' and j[0]!='！' and j[0]!='。' and j[0] != '，' and j[0] != '⋯':            
            seg_pre.append(j)
    seg_X.append(seg_pre)
    list_train.append(seg_pre)
"""for i in range(np.size(val_X)):
    '''seg = emoji.demojize(val_X[i])
    seg = jieba.cut(seg,cut_all=False)'''
    seg = jieba.cut(val_X[i],cut_all=False)
    seg = ' '.join(seg)
    seg = seg.split(' ')
    seg_pre = []
    for j in seg:
        if len(j)!=0 and j[0]!='b' and j[0]!='B' and j[0] not in list_punc and j[0] != '？'\
        and j[0]!='～' and j[0]!='：' and j[0]!='！' and j[0]!='。' and j[0] != '，' and j[0] != '⋯':
            seg_pre.append(j)
    seg_X.append(seg_pre)
    list_val.append(seg_pre)"""
for i in range(np.size(test_X)):
    '''seg = emoji.demojize(test_X[i])
    seg = jieba.cut(seg,cut_all=False)'''
    seg = jieba.cut(test_X[i],cut_all=False)
    seg = ' '.join(seg)
    seg = seg.split(' ')
    seg_pre = []
    for j in seg:
        if j != '' and j[0] not in list_punc and j[0] != '？' and j[0]!='～' and j[0]!='：' and j[0]!='！' and j[0]!='。' and j[0] != '，' and j[0] != '⋯':
          #\and j not in emoji.UNICODE_EMOJI
            seg_pre.append(j)
    seg_X.append(seg_pre)
    list_test.append(seg_pre)
vec_size = 250
max_len = 200 ###沒事不要換
model = Word2Vec(seg_X, size=vec_size, iter=10, sg=1,workers=8)
word_dict = {}
embed_matrix = np.zeros((len(model.wv.vocab) + 1,vec_size))
for i in model.wv.vocab:
    word_dict.update({i:model.wv.vocab[i].index})
    embed_matrix[model.wv.vocab[i].index+1] = model.wv[i]
fp = open('dict4.txt','w')
fp.write(str(word_dict))
fp.close()
np.save('embed_matrix',embed_matrix)
'''vec_size = 250
max_len = 200
embed_matrix = np.load('embed_matrix.npy')
fp = open('dict4.txt','r')
dict_string = fp.read()
word_dict = ast.literal_eval(dict_string)
fp.close()'''
Data_Y = pd.read_csv(sys.argv[2])
array_Y = Data_Y.values
train_Y = array_Y[:,1]
unknown = 0
for i in range(len(list_train)):
    for j in range(len(list_train[i])):
        if list_train[i][j] not in word_dict:
            list_train[i][j] = 0
            unknown += 1
        else:
            list_train[i][j] = word_dict[list_train[i][j]] + 1
train_X = pad_sequences(list_train,maxlen = max_len)#,truncating='post'
train_Y = to_categorical(train_Y).reshape((np.size(train_Y),1,2))
print(unknown)
'''for i in range(len(list_test)):
    for j in range(len(list_test[i])):
        list_test[i][j] = word_dict[list_test[i][j]] + 1
test_X = pad_sequences(list_test,maxlen = max_len,truncating='post')
'''

#model 0
rnn0 = Sequential()
rnn0.add(Embedding(np.size(embed_matrix,0),vec_size,weights=[embed_matrix]))#,input_length=max_len
#rnn.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
#rnn.add(MaxPooling1D(pool_size=2))
rnn0.add(Bidirectional(GRU(256,return_sequences=True)))#recurrent_dropout=0.2
rnn0.add(Bidirectional(GRU(256,return_sequences=True)))#recurrent_dropout=0.2
rnn0.add(TimeDistributed(Dense(256,activation='relu')))
rnn0.add(Dense(128,activation='relu'))
rnn0.add(Dropout(0.1))
rnn0.add(Dense(64,activation='relu'))
rnn0.add(Dropout(0.1))
rnn0.add(Dense(2,activation='softmax'))
rnn0.summary()
#optim = Adam(lr = 0.001)
rnn0.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = []
csvLogger = CSVLogger("log12.csv", separator=",", append=False)
callbacks.append(csvLogger)
modelcheck = ModelCheckpoint('rnn12.h5',monitor='val_acc',verbose=1, save_best_only=True, mode='max', period=1)
callbacks.append(modelcheck)
train_history = rnn0.fit(train_X,train_Y,validation_split=0.1, epochs=15,batch_size=128,\
                          verbose=1, shuffle = False, callbacks = callbacks)
#rnn0.save('rnn0.h5')

#model 1
'''rnn1 = Sequential()
rnn1.add(Embedding(np.size(embed_matrix,0),vec_size,input_length=max_len,weights=[embed_matrix]))
#rnn1.add(Bidirectional(LSTM(128,recurrent_dropout=0.3,dropout=0.3,return_sequences=True)))#wait#wait
rnn1.add(Bidirectional(LSTM(512,recurrent_dropout=0.3,dropout=0.3)))
#rnn.add(Bidirectional(LSTM(256,recurrent_dropout=0.3,dropout=0.3)))#wait
rnn1.add(Dense(256,activation='relu'))
rnn1.add(Dropout(0.5))
rnn1.add(Dense(1,activation='sigmoid'))
rnn1.summary()
#optim = Adam(lr = 0.001)
optim = Adam(lr = 0.0001,decay=1e-6, clipvalue=0.5)
rnn1.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
callbacks = []
csvLogger = CSVLogger("log1.csv", separator=",", append=False)
callbacks.append(csvLogger)
modelcheck = ModelCheckpoint('rnn1.h5',monitor='val_acc',verbose=1, save_best_only=True, mode='max', period=1)
callbacks.append(modelcheck)
train_history = rnn1.fit(train_X,train_Y,validation_data=(val_X,val_Y), epochs=15,batch_size=128,\
                          verbose=1, shuffle = True, callbacks = callbacks)'''
#rnn1.save('rnn1.h5')

#model 2
'''rnn2 = Sequential()
rnn2.add(Embedding(np.size(embed_matrix,0),vec_size,input_length=max_len,weights=[embed_matrix]))
#rnn.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
#rnn.add(MaxPooling1D(pool_size=2))
rnn2.add(Bidirectional(LSTM(128,recurrent_dropout=0.3,dropout=0.3,return_sequences=True)))#wait#wait
rnn2.add(Bidirectional(LSTM(256,recurrent_dropout=0.3,dropout=0.3)))
rnn2.add(Dense(256,activation='relu'))
rnn2.add(Dropout(0.5))
rnn2.add(Dense(1,activation='sigmoid'))
rnn2.summary()
#optim = Adam(lr = 0.001)
optim = Adam(lr = 0.0003,decay=1e-6, clipvalue=0.5)
rnn2.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
callbacks = []
csvLogger = CSVLogger("log2.csv", separator=",", append=False)
callbacks.append(csvLogger)
modelcheck = ModelCheckpoint('rnn2.h5',monitor='val_acc',verbose=1, save_best_only=True, mode='max', period=1)
callbacks.append(modelcheck)
train_history = rnn2.fit(train_X,train_Y,validation_data=(val_X,val_Y), epochs=5,batch_size=128,\
                          verbose=1, shuffle = True, callbacks = callbacks)
rnns.append(rnn0)
#rnn2.save('rnn2.h5')'''

#model 3
'''rnn3 = Sequential()
rnn3.add(Embedding(np.size(embed_matrix,0),vec_size,input_length=max_len,weights=[embed_matrix]))
rnn3.add(LSTM(256,recurrent_dropout=0.3, dropout=0.3))
rnn3.add(Dense(256,activation='relu'))
rnn3.add(Dropout(0.5))
rnn3.add(Dense(1,activation='sigmoid'))
rnn3.summary()
#optim = Adam(lr = 0.001)
optim = Adam(lr = 0.00015,decay=1e-6, clipvalue=0.5)
rnn3.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
callbacks = []
csvLogger = CSVLogger("log3.csv", separator=",", append=False)
callbacks.append(csvLogger)
modelcheck = ModelCheckpoint('rnn3.h5',monitor='val_acc',verbose=1, save_best_only=True, mode='max', period=1)
callbacks.append(modelcheck)
train_history = rnn3.fit(train_X,train_Y,validation_data=(val_X,val_Y), epochs=15,batch_size=128,\
                          verbose=1, shuffle = True, callbacks = callbacks)'''
#rnn3.save('rnn3.h5')

#model 4
'''rnn4 = Sequential()
rnn4.add(Embedding(np.size(embed_matrix,0),vec_size,input_length=max_len,weights=[embed_matrix]))
rnn4.add(LSTM(256,recurrent_dropout=0.3,dropout=0.3,return_sequences=True))
rnn4.add(LSTM(256,recurrent_dropout=0.4,dropout=0.4))
rnn4.add(Dense(256,activation='relu'))
rnn4.add(Dropout(0.5))
rnn4.add(Dense(1,activation='sigmoid'))
rnn4.summary()
#optim = Adam(lr = 0.001)
optim = Adam(lr = 0.0001,decay=1e-6, clipvalue=0.5)
rnn4.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
callbacks = []
csvLogger = CSVLogger("log4.csv", separator=",", append=False)
callbacks.append(csvLogger)
modelcheck = ModelCheckpoint('rnn4.h5',monitor='val_acc',verbose=1, save_best_only=True, mode='max', period=1)
callbacks.append(modelcheck)
train_history = rnn4.fit(train_X,train_Y,validation_data=(val_X,val_Y), epochs=15,batch_size=128,\
                          verbose=1, shuffle = True, callbacks = callbacks)'''
#rnn4.save('rnn4.h5')

'''test_Y = rnn.predict(test_X)
test_Y = np.round(test_Y).astype(int)
f = open('result.csv','w')
f.write('id,label\n')
for i in range(20000):
	 f.write(str(i) + ',' + str(test_Y[i][0]) + '\n')
f.close()'''