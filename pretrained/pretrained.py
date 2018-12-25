import numpy as np

from keras.applications import vgg16
from keras.preprocessing import image

model = vgg16.VGG16()

# Images processed by VGG16 need to be 224 by 224 pixels
unknown_image = image.load_img('bay.jpg', target_size=(224, 224))
unknown_image = image.img_to_array(unknown_image)
unknown_image = np.expand_dims(unknown_image, 0)

# Normalise image's pixel values
unknown_image = vgg16.preprocess_input(unknown_image)

# Run the image through VGG16 model to make a prediction
prediction = model.predict(unknown_image)

# Look up the names of the predicted classes
predicted_classes = vgg16.decode_predictions(prediction, top=9)

print('Top predictions for this image:')

# Get the results for the first image
for _, name, likelihood in predicted_classes[0]:
    print(f"Prediction: {name}, Likelihood: {likelihood:.2f}")
