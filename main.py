from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential
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

# Create a model and add layer
model = Sequential()

# Usually, 2D layers are used. For some kinds of data like sound waves, one
# dimensional layers ared used.
# Create a 2D convolutional layer with 32 different filters to process a 3 by 3
# pixel-sized tile. Each filter will be capable of detecting one pattern in
# the image.
# Add padding just in case the image is not exactly divisible by 3.
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(32, 32, 3), padding='same'))

# Add some extra convolutional layers for extra processing
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Transition from convolutional layers to dense layers with a flatten layer
model.add(Flatten())

# Rectified Linear Units (ReLU) works best with images because it is
# computationally efficient.
# The input_shape is (32, 32, 3) because the images are 32x32 and 3 channels
# for red, green, and blue.
model.add(Dense(512, activation='relu'))

# Since CIFAR-10 data set has 10 different kinds of objects, we create a dense
# layer with 10 nodes. However, when doing classification with more than one
# object, the output layer will usually use a softmax activation function.
model.add(Dense(10, activation='softmax'))

# Print a summary of the model
model.summary()
