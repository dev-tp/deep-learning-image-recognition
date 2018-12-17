from keras.datasets import cifar10
from keras.utils import to_categorical

# Load data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalise data set into continuous (floating point) values
x_test = x_test.astype('float32')
x_test /= 255

x_train = x_train.astype('float32')
x_train /= 255

# Convert class vectors into binary class matrices
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)
