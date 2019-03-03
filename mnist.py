import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
digits = datasets.load_digits()
import matplotlib.pyplot as plt


plt.figure(1,figsize=(3,3))
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


import keras 
import tensorflow as tf

from keras.datasets import mnist
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()


print ("Shape of X_train",X_train.shape)
print("type of X_train",type(X_train))
print ("Shape of X_test",X_test.shape)
print("type of X_test",type(X_test))

X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_train=X_train.astype('float32')
X_train=X_train/255

print("shape of X_train after preprocessing", X_train.shape)
print("type of X_train after preprocessing",X_train.dtype)


X_test=(X_test.reshape(X_test.shape[0],28,28,1).astype('float32'))/255

print("Shape of y_train before preprocessing",Y_train.shape)
Y_train=keras.utils.to_categorical(Y_train,10)
print("shape of y_train after preprocessing",Y_train.shape)
Y_test=keras.utils.to_categorical(Y_test,10)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

model=Sequential()
model.add(Conv2D(filters=30,kernel_size=5, input_shape=(28,28,1),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(15,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))



model.add(Flatten())
model.add(Dense(units=128, activation="relu"))

model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X_train,Y_train, epochs=10, verbose=2)


y_pred=model.predict(X_test)

score=model.evaluate(X_test,Y_test, verbose=0)


