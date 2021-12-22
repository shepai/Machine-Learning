"""
Mandlbrot normalization test
Code by Dexter R Shepherd, aged 20
"""

import numpy as np
import copy
from sklearn import preprocessing
from keras.datasets import mnist

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
