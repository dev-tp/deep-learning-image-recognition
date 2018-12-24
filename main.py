from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
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

# Improve model efficiency by max-pooling. Max-pooling scales down the output of
# the convolutional layers by keeping the largest values and throwing away the
# smaller, least useful values. Typically, max-pooling is performed right after
# a block of convolutional layers.
# Divide the image into 2 by 2 squares and only take the largest value from each
# 2 by 2 region. That will reduce the size of our image while keeping the most
# important values.
model.add(MaxPooling2D(pool_size=(2, 2)))

# Force the neural network to try harder to learn without memorising the input
# data. Randomly throw away some of the data by cutting some of the connections
# between the layers. Dropout layers are usually added after max-pooling layers,
# or after a group of dense layers. Typical dropout percentages range between
# 25% to 50%.
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Transition from convolutional layers to dense layers with a flatten layer
model.add(Flatten())

# Rectified Linear Units (ReLU) works best with images because it is
# computationally efficient.
# The input_shape is (32, 32, 3) because the images are 32x32 and 3 channels
# for red, green, and blue.
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

# Since CIFAR-10 data set has 10 different kinds of objects, we create a dense
# layer with 10 nodes. However, when doing classification with more than one
# object, the output layer will usually use a softmax activation function.
model.add(Dense(10, activation='softmax'))

# Compile model
# Use categorical_crossentropy loss function when trying to classify objects
# into different categories. Otherwise, use binary crossentropy to test if an
# object belongs to one category.
# Use AdAM (Adaptive Momement Estimation) to optimise model.
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

# Print a summary of the model
model.summary()

# Train model
# The batch size is the amount of items we want to feed into the network at once
# during training. If the number is too low, training will take forever and
# might not ever finish. If the number is too high, there's a high chance of
# running out of memory.
# For images, a typical batch size is between 32 and 128 images, but feel free
# to experiment.
# One full pass through the entire training data set is called an epoch. For
# every epoch, the neural network's chances of learning increase; however, it
# will take a long time learn in each pass. At some point, the neural network
# will stop learning, thus it is important to find a value in between. In
# general, the larger your data set, the less epochs you need to train.
# Shuffle data batches to prevent the order from influencing training results.

# Loss is the numerical representation of how wrong the model is, while accuracy
# represents how often the model is making the correct predition for the
# training data. If the loss doesn't go down over time, the accuracy will not
# improve. In that case, something may be wrong with the neural network design,
# or there are problems with the training data. It may also be that the data set
# may be too small to train a neural network, or that the model doesn't have
# enough layers to capture the patterns in the data set.
model.fit(x_train, y_train, batch_size=64, epochs=30,
          shuffle=True, validation_data=(x_test, y_test))

# Save neural network structure; this can be reused with different weights and
# other inputs.
with open('model_structure.json', 'w') as json_file:
    json_file.write(model.to_json())

# Save neural network's trained weights
model.save_weights('model_weights.h5')
