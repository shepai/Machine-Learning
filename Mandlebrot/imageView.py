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

def mandelbrot(height, width, x_from=0, x_to=0.1, y_from=0, y_to=1, max_iterations=100):
    x = np.linspace(x_from, x_to, width).reshape((1, width))
    y = np.linspace(y_from, y_to, height).reshape((height, 1))
    c = x + 1j * y
    return c
C=mandelbrot(28,28)

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

w = 28
h = 28
columns = 3
rows = 1
for i in range(10):
    fig = plt.figure()
    fig.add_subplot(rows, columns, 1)
    plt.imshow(train_X[i])
    fig.add_subplot(rows, columns, 2)
    plt.imshow(X_standard_train[i].reshape(28,28))
    fig.add_subplot(rows, columns, 3)
    plt.imshow(X_mandle_train[i])
    plt.show()

