#import libraries
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

#load the data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
# print(y_train[0])
#y_train = y_train.astype('float32')
#y_test = y_test.astype('float32')

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# Build the architecture

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))