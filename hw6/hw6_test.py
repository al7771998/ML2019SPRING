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
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model
import emoji
import ast
jieba.load_userdict(sys.argv[2])
Data_X = pd.read_csv(sys.argv[1])
array_X = Data_X.values
test_X = array_X[:,1]
list_test = []
list_punc = string.punctuation
for i in range(np.size(test_X)):
    seg = jieba.cut(test_X[i],cut_all=False)
    seg = ' '.join(seg)
    seg = seg.split(' ')
    seg_pre = []
    for j in seg:
        if j != '' and j[0] not in list_punc and j[0] != '？' and j[0]!='～' and j[0]!='：' and j[0]!='！' and j[0]!='。' and j[0] != '，' and j[0] != '⋯':            
        #\and j not in emoji.UNICODE_EMOJI
            seg_pre.append(j)
    #if len(seg) > max_len:
    list_test.append(seg_pre)
max_len = 250
unknown = 0
fp = open('dict4.txt','r')
dict_string = fp.read()
word_dict = ast.literal_eval(dict_string)
fp.close()
rnn = load_model('rnn12.h5')
for i in range(len(list_test)):
    for j in range(len(list_test[i])):
        if list_test[i][j] not in word_dict:
            list_test[i][j] = 0
            unknown += 1
        else:
            list_test[i][j] = word_dict[list_test[i][j]] + 1
print(unknown)
test_X = pad_sequences(list_test,maxlen = max_len)
rnn.summary()
test_Y = rnn.predict_classes(test_X)
print(test_Y)
f = open(sys.argv[3],'w')
f.write('id,label\n')
for i in range(20000):
	 f.write(str(i) + ',' + str(test_Y[i][0]) + '\n')
f.close()