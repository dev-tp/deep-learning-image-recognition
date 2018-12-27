import joblib
import matplotlib.image as mpimg
import numpy as np
import os

from keras.applications import vgg16

images = []
labels = []

for file_name in os.listdir('not-dogs'):
    images.append(mpimg.imread(f"not-dogs/{file_name}"))
    labels.append(0)

for file_name in os.listdir('dogs'):
    images.append(mpimg.imread(f"dogs/{file_name}"))
    labels.append(1)

x_train = np.array(images)
y_train = np.array(labels)

del labels, images

# Normalise data
x_train = vgg16.preprocess_input(x_train)

# Use VGG16 for feature extraction. Create an instance of VGG16 with
# include_top=False to remove the top (last) layer from the neural network.
# Set the input_shape to (64, 64, 3) to process 64 by 64 images and its three
# colour layers (RBG).
# weights defaults to imagenet, but in case it changes in the future, it is
# included.
pretrained_neural_net = vgg16.VGG16(
    include_top=False, input_shape=(64, 64, 3), weights='imagenet')

# Extract features for each image (all in one pass)
features_x = pretrained_neural_net.predict(x_train)

# Dump arrays to output files
joblib.dump(features_x, 'x-train.dat')
joblib.dump(y_train, 'y-train.dat')
