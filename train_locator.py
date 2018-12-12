# Tensorflow & Keras
from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow import data
from tensorflow.contrib.saved_model import save_keras_model

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# import sys
# sys.path.append("..")

import Utils.Utils as utils

# Model
from Locator.model import create_model

# FaceLandmark Dataset
from Database.FaceLandmark.Database import get_data
# from .. import Database.FaceLandmark.Database.get_data

X, Y = get_data(image_path='Database/FaceLandmark/face_images.npz',
                          landmark_path='Database/FaceLandmark/facial_keypoints.csv')

# Split into train and test
Xtrain, Ytrain, Xtest, Ytest = utils.split_set(X, Y, 0.2)

print('Xtrain.shape = {}'.format(Xtrain.shape))
print('Xtest.shape = {}'.format(Xtest.shape))

# Use tensorflow Dataset function
training_set = data.Dataset.from_tensor_slices((Xtrain, Ytrain))
training_set = training_set.batch(32).repeat()
valid_set = data.Dataset.from_tensor_slices((Xtest, Ytest))
valid_set = valid_set.batch(32).repeat()

input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)

print('input_shape = {}'.format(input_shape))

model = create_model((input_shape))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

# model.fit(Xtrain, Ytrain, batch_size=128, epochs=10, validation_split=0.2, verbose = 1)

# Callbacks
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./Logs')
]

model.fit(
    training_set.make_one_shot_iterator(),
    epochs=10,
    steps_per_epoch=30,
    validation_data=valid_set,
    validation_steps=3,
    verbose=1,
    callbacks=callbacks)

# Save keras model
model.save('Models/model_locator.h5')

# Save tensorflow checkpoint
# output_path = save_keras_model(model, 'Models/model_locator')

# print('Tensorflow model saved path : '.format(output_path))