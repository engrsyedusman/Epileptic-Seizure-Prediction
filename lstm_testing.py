# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:47:50 2020

@author: Syed Usman
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:31:47 2020

@author: Syed Usman
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 01:31:21 2019

@author: Syed Usman
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:11:10 2019

@author: Syed Usman
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import numpy as np
#from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pylab as plt
from keras.utils import np_utils
from keras.layers import LSTM
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split

#Load Images
X = np.load("data.npy")
#Data is 2062 x 64 x 64 matrix
print(X.shape)
#Load labels - One hot encoded: 2062 x 10
Y = np.load("labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state= 1234)

"""
x_train=X[0:96,:,:,:];
y_train=Y[0:96,:];
x_test=X[97:193,:,:,:];
y_test=Y[97:193,:];
"""
#y_train=np_utils.to_categorical(y_train)
#print(Y.shape)
#x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.50,random_state=42)
#x_train=X;
#y_train=Y;
np.random.seed(123)
#img_x,img_y=16,400
#x_train=x_train.reshape(x_train.shape[0],img_x,img_y,1)
#x_test=x_test.reshape(x_test.shape[0],img_x,img_y,1)
#input_shape=(img_x,img_y,1)
#x_train=x_train.astype('float32')
#x_test=x_test.astype('float32')
#x_train/=255
#x_test/=255
num_classes=2
print(X_train.shape)
#y_train=to_categorical(y_train,2)
#y_test=to_categorical(y_test,2)
print(X_test.shape)
# Convolution Layer 1
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', input_shape = (65,117,23)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(BatchNormalization())


#Convolution Layer 2
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(BatchNormalization())

#Convolution Layer 3
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(BatchNormalization())
model.add(Flatten())
#model.add(Dense(650, activation='sigmoid'))


# Adding SVM Layers
from keras import backend as K
for l in range(len(model.layers)):
    print(l, model.layers[l])
    
    

# feature extraction layer
getFeature = K.function([model.layers[0].input, K.learning_phase()],
                        [model.layers[12].output])
X_train = getFeature([X_train, 0])[0]
X_test = getFeature([X_test, 0])[0]



X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SimpleRNN

# code for building an LSTM with 100 neurons and dropout. Runs for 50 epochs

model2 = Sequential()
#model.add(Embedding(71680, 32, input_length=7168))
model2.add(LSTM(256, input_shape=(1,7168)))
#model.add(LSTM(128, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(128))
#model.add(LSTM(128, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(Dense(30, activation='relu'))


#model.add(LSTM(32, return_sequences=False, input_shape=(7168,1)))
#model.add(Dropout(0.5))
#model.add(LSTM(100)) dramatically worse results
model2.add(Dense(2, activation='softmax'))

model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#y_train=np.expand_dims(y_train, 2)
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
model2.fit(X_train, y_train, batch_size=32, epochs=10)
score = model2.evaluate(X_test, y_test, batch_size=16)
y_pred=model.predict(X_test)





from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  






"""
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
X=X_train
Y=y_train
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
#results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
#print(results.mean())
y_train=y_train.ravel()
y_test=y_test.ravel()
ensemble.fit(X_train,y_train)
y_pred=ensemble.predict(X_test)
TP=0
TN=0
FP=0
FN=0
for i in range(0,91):
    if y_test[i]==1:
        if y_pred[i]==y_test[i]:
            TP+=1
        else:
            FN+=1
    else:
        if y_pred[i]==y_test[i]:
            TN+=1
        else:
            FP+=1
Accuracy=(TP+TN)/(TP+TN+FP+FN)
Sensitivity=TP/(TP+FN)
Specificity=TN/(TN+FP)


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  

"""