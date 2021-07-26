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
model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', activation = 'relu', input_shape = (65,117,23)))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(BatchNormalization())


#Convolution Layer 2
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(BatchNormalization())

#Convolution Layer 3
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation = 'relu'))
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
                        [model.layers[9].output])
X_train = getFeature([X_train, 0])[0]
X_test = getFeature([X_test, 0])[0]


"""
# Training SVM and computing confusion matrix
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)
print("Fitting done")
y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  

"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:56:05 2020

@author: Syed Usman
"""


#PCA for feature reduction


from sklearn.decomposition import PCA
pca = PCA(n_components=90)
traindata = pca.fit_transform(traindata)
testdata = pca.fit_transform(testdata)







# Features visualization
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[10].reshape(1,65,117,23))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1


plt.imshow(X_train[10][:,:,1]);
display_activation(activations, 4, 4, 1)

# Fully Connected Layers
# FC layer 1
#model.add(Flatten())
#
#model.add(Dense(256, activation='sigmoid'))
#
## FC layer 2
#
#model.add(Dense(2, activation='softmax'))


#model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=["accuracy"])
#history=model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=1)

#score = model.evaluate(x_test, y_test, verbose=1)
#print('Test Loss:',score[0])

#matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
"""
model.add(Conv2D(filters = 16, kernel_size = (16,400),padding = 'Same', activation = 'relu', input_shape = (16,400,1)))
#model.add(BatchNormalization())
#model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', activation = 'relu'))
#model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))
#model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(filters = 10, kernel_size = (4,4),padding = 'Same', activation = 'relu'))
#model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
#model.add(Conv2D(filters = 10, kernel_size = (4,4),padding = 'Same', activation = 'relu'))
#model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(256, activation='sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=["accuracy"])
history=model.fit(x_train,y_train,batch_size=128,epochs=10,verbose=1)
print(x_train.shape)
print(x_test.shape)
score = model.evaluate(x_test, y_test, verbose=1)
print('Test Loss:',score[0])
print('Test accuracy:',score[1])
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend(['Accuracy','Loss'],loc='upper left')
#matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
"""