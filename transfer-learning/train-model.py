import joblib

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# Load data
x_train = joblib.load('x-train.dat')
y_train = joblib.load('y-train.dat')

# Create model and add layers
model = Sequential()

# No convolutional layers are added because the model is based on VGG16
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=[
              'accuracy'], optimizer='adam')

model.fit(x_train, y_train, epochs=10, shuffle=True)

with open('model-structure.json', 'w') as json_file:
    json_file.write(model.to_json())

model.save_weights('model-weights.h5')
