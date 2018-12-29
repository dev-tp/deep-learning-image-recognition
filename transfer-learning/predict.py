import numpy as np

from keras.applications import vgg16
from keras.models import model_from_json
from keras.preprocessing import image

json_file = open('model-structure.json', 'r')
model = model_from_json(json_file.read(-1))
json_file.close()

model.load_weights('model-weights.h5')

unknown_image = image.load_img('image.png', target_size=(64, 64))
unknown_image = image.img_to_array(unknown_image)

unknown_images = np.expand_dims(unknown_image, 0)
unknown_images = vgg16.preprocess_input(unknown_images)

feature_extration_model = vgg16.VGG16(
    include_top=False, input_shape=(64, 64, 3), weights='imagenet')
features = feature_extration_model.predict(unknown_images)

results = model.predict(features)
first_result = results[0][0]

print(f"Likelihood that the image contains a dog: {first_result * 100:.1f}%")
