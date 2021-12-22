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
 
x = np.linspace(-2, 1, num=m).reshape((1, m))
y = np.linspace(-1, 1, num=n).reshape((n, 1))
C = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))

"""
Normalize datasets
"""
#perform preprocessing scaling of the data
scaler1 = preprocessing.StandardScaler().fit(train_X)
X_standard_train = scaler1.transform(train_X)
scaler2 = preprocessing.StandardScaler(test_X).fit(test_X)
X_standard_test = scaler2.transform()

#gather mandlebrot normalizartion
X_mandle_train = []
X_mandle_test = []

for im in train_X:
    X_mandle_train.append(np.dot(C,im))
for im in test_X:
    X_mandle_test.append(np.dot(C,im))

"""
Create models
"""
class model:
    def __init__(self):
        self.model=Sequential() #2 by 64 neural network
        self.model.add(Conv2D(64,(3,3),input_shape=28*28))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(64,(3,3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))

        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
    def train(self,x,y,epochs=30):
        history=self.model.fit(x,y,batch_size=32,epochs=epochs,validation_split=0.1)
        self.history = history.history #gather training log
    def test(self,X,y):
        assert len(X)==len(y), "Error, the arrays do not match length"
        predictions = model.predict(X)
        count=0
        for i in range(len(predictions)):
            if y[i]==predictions[i]:
                count+=1
        return count/len(predictions)

model_plain=model()
model_standard=model()
model_mandle=model()

"""
Train models
"""
model_plain.train(train_X,train_y)
model_standard.train(X_standard_train,train_y)
model_mandle.train(X_mandle_train,train_y)

"""
Test models
"""
p1=model_plain.test(test_X,test_y)
p2=model_standard.test(X_standard_test,test_y)
p3=model_mandle.test(X_mandle_test,test_y)

print(p1,p2,p3)