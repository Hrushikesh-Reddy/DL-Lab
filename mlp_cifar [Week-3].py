from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

#dont write in record
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Import data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the architecture

model = Sequential()
model.add(Flatten(Input_shape=(32, 32, 3)))