import numpy as np

from keras.models import model_from_json
from keras.preprocessing import image

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

json_file = open('model-structure.json')

# Reload model structure
model = model_from_json(json_file.read(-1))
model.summary()

json_file.close()

# Load weights
model.load_weights('model-weights.h5')

# Load image, shrink to 32 by 32 pixels, and convert image to nparray
unknown_image = image.img_to_array(
    image.load_img('image.png', target_size=(32, 32)))

# Keras processes data in batches. Since only one image was loaded, call
# np.expand_dims to append it to a new list.
images = np.expand_dims(unknown_image, 0)

# Get first result from results since only one image was passed
result = model.predict(images)[0]

# result is an array of weights mapped to each of the CIFAR-10 class names
most_likely_class_index = int(np.argmax(result))

likelihood = result[most_likely_class_index]
label = cifar10_class_names[most_likely_class_index]

print(f"The model predicts the image is a(n) {label} with a likelihood of "
      f"{likelihood:.2f}.")
