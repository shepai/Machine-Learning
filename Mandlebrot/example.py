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

def mandelbrot(height, width, x_from=0, x_to=1, y_from=0, y_to=1, max_iterations=100):
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


"""
Reshape data for neural network
"""
train_X=train_X.reshape(len(train_X),28*28)
test_X=test_X.reshape(len(test_X),28*28)
X_standard_train=X_standard_train.reshape(len(X_standard_train),28*28)
X_standard_test=X_standard_test.reshape(len(X_standard_test),28*28)
X_mandle_train=X_mandle_train.reshape(len(X_mandle_train),28*28)
X_mandle_test=X_mandle_test.reshape(len(X_mandle_test),28*28)

train_y_ = np.zeros((train_y.size, train_y.max()+1))
train_y_[np.arange(train_y.size),train_y] = 1

test_y_ = np.zeros((test_y.size, test_y.max()+1))
test_y_[np.arange(test_y.size),test_y] = 1

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


if os.path.isdir("./plain"):
    model_plain.model = tf.keras.models.load_model('./plain')
else:
    model_plain.train(train_X,train_y_)
    model_plain.save("plain")
if os.path.isdir("./standard"):
    model_standard.model = tf.keras.models.load_model('./standard')
else:
    model_standard.train(X_standard_train,train_y_)
    model_standard.save("standard")
if os.path.isdir("./mandle"):
    model_mandle.model = tf.keras.models.load_model('./mandle')
else:
    model_mandle.train(X_mandle_train,train_y_)
    model_mandle.save("mandle")

Test models

p1=model_plain.test(test_X,test_y)
p2=model_standard.test(X_standard_test,test_y)
p3=model_mandle.test(X_mandle_test,test_y)

print(p1,p2,p3)
"""

"""
Experiments
"""
#1 find how unprocessed data effects the models
percent=0.4

model_plain.train(train_X,train_y_,epochs=25)
model_standard.train(X_standard_train,train_y_,epochs=25)
model_mandle.train(X_mandle_train,train_y_,epochs=25)

datasets=[(test_X,test_y),(X_standard_test,test_y),(X_mandle_test,test_y)]
data=[]
for train,test in datasets: #loop through different sized epochs
    #train based on epoch
    #test
    t=[]
    t.append(model_plain.test(train,test))
    t.append(model_standard.test(train,test))
    t.append(model_mandle.test(train,test))
    data.append(t)
langs = ['Unprocessed dataset','Gaussian dataset','Mandelbrot dataset']
data=np.array(data)
X_axis = np.arange(len(langs))
print(sum(data[:, 0])/3)
print(sum(data[:, 1])/3)
print(sum(data[:, 2])/3)
plt.bar(X_axis, data[:, 0], 0.2, label = 'Unprocessed')
plt.bar(X_axis + 0.2, data[:, 1], 0.2, label = 'Gaussian')
plt.bar(X_axis + 0.4, data[:, 2], 0.2, label = 'Mandelbrot')

plt.xticks(X_axis, langs)
plt.title("How models perform on alternative datasets")
plt.ylabel("Accuracy")
plt.xlabel("Size of batch")
plt.legend(loc="upper right")
plt.show()

#2 find how training batch size affects the perforance
plainData=[]
standardData=[]
mandleData=[]
percent=0.4
for i in range(5,50,5): #loop through different sized epochs
    model_plain=model()
    model_standard=model()
    model_mandle=model()
    #train based on epoch
    model_plain.train(train_X[0:int(len(train_X)*percent)],train_y_[0:int(len(train_X)*percent)],epochs=10,batch=i)
    model_standard.train(X_standard_train[0:int(len(X_standard_train)*percent)],train_y_[0:int(len(X_standard_train)*percent)],epochs=10,batch=i)
    model_mandle.train(X_mandle_train[0:int(len(X_mandle_train)*percent)],train_y_[0:int(len(X_mandle_train)*percent)],epochs=10,batch=i)
    #test
    plainData.append(model_plain.test(test_X,test_y))
    standardData.append(model_standard.test(X_standard_test,test_y))
    mandleData.append(model_mandle.test(X_mandle_test,test_y))

plt.plot([i for i in range(5,50,5)],plainData,'k--', label='No preprocessing',c="g")
plt.plot([i for i in range(5,50,5)],standardData,'k:', label='Gaussian preprocessing',c="r")
plt.plot([i for i in range(5,50,5)],mandleData,'k', label='Mandelbrot preprocessing',c="b")
plt.title("Batch size")
plt.ylabel("Accuracy")
plt.xlabel("Size of batch")
plt.legend(loc="lower right")
plt.show()

#3 find how training data effects the perforance
plainData=[]
standardData=[]
mandleData=[]

for percent in range(10,100,10): #loop through different sized epochs
    percent/=100 #make percentage
    model_plain=model()
    model_standard=model()
    model_mandle=model()
    #train based on epoch
    model_plain.train(train_X[0:int(len(train_X)*percent)],train_y_[0:int(len(train_X)*percent)],epochs=10)
    model_standard.train(X_standard_train[0:int(len(X_standard_train)*percent)],train_y_[0:int(len(X_standard_train)*percent)],epochs=10)
    model_mandle.train(X_mandle_train[0:int(len(X_mandle_train)*percent)],train_y_[0:int(len(X_mandle_train)*percent)],epochs=10)
    #test
    plainData.append(model_plain.test(test_X,test_y))
    standardData.append(model_standard.test(X_standard_test,test_y))
    mandleData.append(model_mandle.test(X_mandle_test,test_y))

plt.plot([i*10 for i in range(1,len(plainData)+1)],plainData,'k--', label='No preprocessing',c="g")
plt.plot([i*10 for i in range(1,len(standardData)+1)],standardData,'k:', label='Gaussian preprocessing',c="r")
plt.plot([i*10 for i in range(1,len(mandleData)+1)],mandleData,'k', label='Mandelbrot preprocessing',c="b")
plt.title("Size of data vs accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Percentage of dataset used for training")
plt.legend(loc="lower right")
plt.show()

#4 find how epochs affect the performance
plainData=[]
standardData=[]
mandleData=[]
percent=0.2
for epoch in range(30): #loop through different sized epochs
    model_plain=model()
    model_standard=model()
    model_mandle=model()
    #train based on epoch
    model_plain.train(train_X[0:int(len(train_X)*percent)],train_y_[0:int(len(train_X)*percent)],epochs=epoch)
    model_standard.train(X_standard_train[0:int(len(train_X)*percent)],train_y_[0:int(len(train_X)*percent)],epochs=epoch)
    model_mandle.train(X_mandle_train[0:int(len(train_X)*percent)],train_y_[0:int(len(train_X)*percent)],epochs=epoch)
    #test
    plainData.append(model_plain.test(test_X,test_y))
    standardData.append(model_standard.test(X_standard_test,test_y))
    mandleData.append(model_mandle.test(X_mandle_test,test_y))

plt.plot([i for i in range(len(plainData))],plainData,'k--', label='No preprocessing',c="g")
plt.plot([i for i in range(len(standardData))],standardData,'k:', label='Gaussian preprocessing',c="r")
plt.plot([i for i in range(len(mandleData))],mandleData,'k', label='Mandelbrot preprocessing',c="b")
plt.title("The role of epochs to determining the accuracy of preprocessing methods on unseen data")
plt.ylabel("Accuracy")
plt.xlabel("Epoch number")
plt.legend(loc="lower right")
plt.show()
