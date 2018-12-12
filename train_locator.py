# Tensorflow & Keras
from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.contrib.saved_model import save_keras_model

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# import sys
# sys.path.append("..")

# Model
from Locator.model import create_model

# Dataset
from Database.FaceLandmark.Database import get_data
# from .. import Database.FaceLandmark.Database.get_data

Xtrain, Ytrain = get_data(image_path='Database/FaceLandmark/face_images.npz',
                          landmark_path='Database/FaceLandmark/facial_keypoints.csv')

input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)

print('input_shape = {}'.format(input_shape))

model = create_model((input_shape))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(Xtrain, Ytrain, batch_size=128, epochs=1, validation_split=0.2, verbose = 1)

# Save keras model
model.save('Models/model_locator.h5')

# Save tensorflow checkpoint
output_path = save_keras_model(model, 'Models/model_locator')

print('Tensorflow model saved path : '.format(output_path))