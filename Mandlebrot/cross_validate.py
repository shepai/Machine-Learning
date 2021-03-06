"""
Mandlbrot normalization test
Code by Dexter R Shepherd, aged 20
"""

import numpy as np
import copy
from sklearn import preprocessing
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
from matplotlib import pyplot as plt
"""
Gather data from MNIST
"""
(train_X, train_y), (test_X, test_y) = mnist.load_data() ##load data
#display the shapes
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

"""
Create mandlebrot
"""

m = train_X.shape[1]
n = train_X.shape[2]

def mandelbrot(height, width, x_from=0, x_to=0.1, y_from=0, y_to=0.1, max_iterations=100):
    x = np.linspace(x_from, x_to, width).reshape((1, width))
    y = np.linspace(y_from, y_to, height).reshape((height, 1))
    c = x + 1j * y
    return c
C=mandelbrot(1,28*28).reshape(28,28)

"""
Normalize datasets
"""
#perform preprocessing scaling of the data
toTrain=train_X.reshape(len(train_X),28*28)
toTest=test_X.reshape(len(test_X),28*28)

scaler1 = preprocessing.StandardScaler().fit(toTrain)
X_standard_train = scaler1.transform(toTrain)
scaler2 = preprocessing.StandardScaler().fit(toTest)
X_standard_test = scaler2.transform(toTest)

#gather mandlebrot normalizartion
X_mandle_train = []
X_mandle_test = []

for im in train_X:
    X_mandle_train.append(np.multiply(C,im))
for im in test_X:
    X_mandle_test.append(np.multiply(C,im))
#convert to numpy
X_mandle_train=np.array(X_mandle_train)
X_mandle_test=np.array(X_mandle_test)
X_mandle_train=X_mandle_train.astype(float)
X_mandle_test=X_mandle_test.astype(float)

#randomize data
p=np.random.permutation(len(train_y))
X_mandle_train=X_mandle_train[p]
train_y1=train_y[p]
p=np.random.permutation(len(test_y))
X_mandle_test=X_mandle_test[p]
test_y1=test_y[p]

p=np.random.permutation(len(train_y))
X_standard_train=X_standard_train[p]
train_y2=train_y[p]
p=np.random.permutation(len(test_y))
X_standard_test=X_standard_test[p]
test_y2=test_y[p]

p=np.random.permutation(len(train_y))
train_X=train_X[p]
train_y3=train_y[p]
p=np.random.permutation(len(test_y))
test_X=test_X[p]
test_y3=test_y[p]

"""
Reshape data for neural network
"""
train_X=train_X.reshape(len(train_X),28*28)
test_X=test_X.reshape(len(test_X),28*28)
X_standard_train=X_standard_train.reshape(len(X_standard_train),28*28)
X_standard_test=X_standard_test.reshape(len(X_standard_test),28*28)
X_mandle_train=X_mandle_train.reshape(len(X_mandle_train),28*28)
X_mandle_test=X_mandle_test.reshape(len(X_mandle_test),28*28)

train_y_1 = np.zeros((train_y1.size, train_y1.max()+1))
train_y_1[np.arange(train_y1.size),train_y1] = 1

test_y_2 = np.zeros((test_y2.size, test_y2.max()+1))
test_y_2[np.arange(test_y2.size),test_y2] = 1

train_y_3 = np.zeros((train_y3.size, train_y3.max()+1))
train_y_3[np.arange(train_y3.size),train_y3] = 1

test_y_3 = np.zeros((test_y3.size, test_y3.max()+1))
test_y_3[np.arange(test_y3.size),test_y3] = 1

train_y_2 = np.zeros((train_y2.size, train_y2.max()+1))
train_y_2[np.arange(train_y2.size),train_y2] = 1

test_y_1 = np.zeros((test_y1.size, test_y1.max()+1))
test_y_1[np.arange(test_y1.size),test_y1] = 1

"""
Create models
"""
class model:
    def __init__(self,outcomes=10,numCells=28*28):
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=numCells, activation='relu')) #4000 inputs compressed to 200
        #model.add(Dense(100, activation='relu')) #keep abstracting information
        #model.add(Dense(10, activation='relu'))
        self.model.add(Dense(outcomes, activation='sigmoid')) #sigmoid is good for binary
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self,x,y,epochs=30,batch=32):
        history=self.model.fit(x,y,batch_size=batch,epochs=epochs,validation_split=0.1)
        self.history = history.history #gather training log

    def test(self,X,y):
        assert len(X)==len(y), "Error, the arrays do not match length"
        predictions = self.model.predict(X)
        count=0

        for i in range(len(predictions)):
            pred=np.argmax(predictions[i])
            if y[i]==pred:
                count+=1
        return count/len(predictions)
    def save(self,name):
        self.model.save(''+name)


model_plain=model()
model_standard=model()
model_mandle=model()

"""
Train models
"""

model_plain.train(train_X,train_y_3,epochs=10)
model_standard.train(X_standard_train,train_y_2,epochs=10)
model_mandle.train(X_mandle_train,train_y_1,epochs=10)

"""
Test models
"""

p1=model_plain.test(test_X,test_y3)
p2=model_standard.test(X_standard_test,test_y2)
p3=model_mandle.test(X_mandle_test,test_y1)

print("Normal:",p1,"\nGaussian:",p2,"\nMandelbrot:",p3)
