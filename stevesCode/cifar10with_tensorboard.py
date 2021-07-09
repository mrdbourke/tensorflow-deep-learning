# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:49:06 2021
https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/
@author: steve
"""

import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from time import time

# Model configuration
img_width, img_height = 32, 32
batch_size = 250
no_epochs = 2
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load CIFAR10 dataset
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Visualize CIFAR10 dataset
import matplotlib.pyplot as plt
classes = {
  0: 'airplane',
  1: 'automobile',
  2: 'bird',
  3: 'cat',
  4: 'deer',
  5: 'dog',
  6: 'frog',
  7: 'horse',
  8: 'ship',
  9: 'truck'
}
fig, axes = plt.subplots(2,5, sharex=True)
axes[0,0].imshow(input_train[0])
axes[0,1].imshow(input_train[1])
axes[0,2].imshow(input_train[2])
axes[0,3].imshow(input_train[3])
axes[0,4].imshow(input_train[4])
axes[1,0].imshow(input_train[5])
axes[1,1].imshow(input_train[6])
axes[1,2].imshow(input_train[7])
axes[1,3].imshow(input_train[8])
axes[1,4].imshow(input_train[9])
axes[0,0].set_title(classes[target_train[0][0]])
axes[0,1].set_title(classes[target_train[1][0]])
axes[0,2].set_title(classes[target_train[2][0]])
axes[0,3].set_title(classes[target_train[3][0]])
axes[0,4].set_title(classes[target_train[4][0]])
axes[1,0].set_title(classes[target_train[5][0]])
axes[1,1].set_title(classes[target_train[6][0]])
axes[1,2].set_title(classes[target_train[7][0]])
axes[1,3].set_title(classes[target_train[8][0]])
axes[1,4].set_title(classes[target_train[9][0]])
axes[0,0].set_axis_off()
axes[0,1].set_axis_off()
axes[0,2].set_axis_off()
axes[0,3].set_axis_off()
axes[0,4].set_axis_off()
axes[1,0].set_axis_off()
axes[1,1].set_axis_off()
axes[1,2].set_axis_off()
axes[1,3].set_axis_off()
axes[1,4].set_axis_off()
plt.show()

# Set input shape
input_shape = (img_width, img_height, 3)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Convert them into black or white: [0, 1].
input_train = input_train / 255
input_test = input_test / 255

# Convert target vectors to categorical targets
target_train = tensorflow.keras.utils.to_categorical(target_train, no_classes)
target_test = tensorflow.keras.utils.to_categorical(target_test, no_classes)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Define Tensorboard as a Keras callback
tensorboard = TensorBoard(log_dir='.\logs', histogram_freq=1, 
                          write_images=True, write_grads=False)
keras_callbacks = [  tensorboard]

# Fit data to model
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split,
          callbacks=keras_callbacks)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')#----------------------------------------------------------------------