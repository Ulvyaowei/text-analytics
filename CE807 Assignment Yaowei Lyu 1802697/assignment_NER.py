#!/usr/bin/python
# _*_ coding:utf-8 _*_

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report,accuracy_score

#format the data
raw_data = open('aij-wikiner-en-wp2','r').readlines()
train_data = open('temp.txt','w')

update_data = ''

for lines in raw_data:
    clean_data = lines.strip().split(' ')
    for i in clean_data:
        part_data = i.split('|')
        if (len(part_data)>1):
            del part_data[1]
            update_data = part_data[0]+' '+part_data[1]
        train_data.write(update_data)
        train_data.write('\n')
    train_data.write('\n')
train_data.close()


#get the raw data
raw_train_data = pd.read_csv('temp.txt',sep = ' ',delimiter=' ',header=None)
raw_train_data.columns = ['word','IOB']
train_data = raw_train_data.iloc[:,[0,1]]
train_data = train_data[:20000]
trainword_appear_time = train_data.word.nunique()
trainIOB_appear_time = train_data.IOB.nunique()
#let the data get into group
## 1
train_data.groupby('IOB').size().reset_index(name = 'index')
## 2
train_data_word = train_data.drop('IOB',axis = 1)
train_data_IOB = train_data.IOB.values


raw_test_data = pd.read_csv('wikigold.conll.txt',sep='\n',delimiter=' ',header=None)
raw_test_data.columns = ['word','IOB']
test_data = raw_test_data.iloc[:,[0,1]]
test_data = test_data[:20000]
testword_appear_time = train_data.word.nunique()
testIOB_appear_time = train_data.IOB.nunique()
test_data.groupby('IOB').size().reset_index(name = 'index')

test_data_word = raw_test_data.drop('IOB',axis = 1)
#transfor word into vector
vector = DictVectorizer(sparse=False)
train_data_word = vector.fit_transform(train_data_word.to_dict('records'))
test_data_word = vector.transform(test_data_word.to_dict('records'))
train_data_IOB = test_data.IOB.values
#split the data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(train_data_word,train_data_IOB,test_size = 0.20,random_state=0)
per_model = Perceptron(verbose=10, n_jobs=-1, max_iter=5)

final_model = per_model.fit(X_train,Y_train)

print('Performance after train with the training data :')
print(classification_report(y_pred=final_model.predict(X_test),y_true=Y_test))
print('Accuracy:',accuracy_score(Y_test,final_model.predict(X_test)))


