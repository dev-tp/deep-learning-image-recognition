import matplotlib.pyplot as plt
from keras.datasets import cifar10

cifar10_class_names = [
    'Plane',
    'Car',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Boat',
    'Truck',
]

(x_train, y_train), _ = cifar10.load_data()

for i in range(10):
    sample_image = x_train[i]
    sample_image_class_name = cifar10_class_names[y_train[i][0]]

    plt.imshow(sample_image)
    plt.title(sample_image_class_name)

    plt.show()
