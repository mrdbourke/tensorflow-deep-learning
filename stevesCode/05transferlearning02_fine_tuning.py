# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:06:12 2021
Most important takeaways:... 
    1. image_dataset_from_directory: Faster, simpler
    2. Functional API
@author: steve
"""

# %% Imports...
import numpy as np, matplotlib.pyplot as plt, os as os, pandas as pd, seaborn as sns
# 1 July 2021... next two statements...BEFORE any tensorflow did the trick.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time , sys, tensorflow as tf, tensorboard, sklearn.metrics, itertools, io

import random

from datetime import datetime
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers.experimental import preprocessing

# import pydot
import graphviz  #for graph of 
import GPUtil
gpus = GPUtil.getGPUs()
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import tensorPrepStarter as tps

!nvidia-smi

# %% get data
# url = url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip'
# zipfilepath = 'd:/data/udemy/dbourkeTFcert'
# tps.unzip_data(zipfilepath, url) 
# %% tps.ShowImageFolder_SetTrainTestDirs
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_10_percent'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

# %% Create the data loaders
# normalize 0 to 255; turn data into "Flows of batches of data"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SHAPE= (224, 224)  # HYPER PARAMETER
BATCH_SIZE = 32

# the following "Image datagen INSTANCE" is not needed in the new function below...
# train_datagen = ImageDataGenerator(rescale=1/255.)
# test_dategen  = ImageDataGenerator(rescale=1/255.)

# below, we are using "image_dataset_from_directory" ...this is faster, simpler (no datagen instance needed)
train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it
test_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(directory=test_dir, image_size=IMAGE_SHAPE,
                                 label_mode='categorical', batch_size=BATCH_SIZE
                                 ,seed=42)
#                BATCH!!                     IMAGES         LABELS, ONE-HOT ENCODED
#train_batch = <BatchDataset shapes: ((None, 224, 224, 3), (None, 10)), types: (tf.float32, tf.float32)>
# notice you do lots of thing (methods)
# Let's look at what we've made with the 'train_batches'...
train_batch.class_names
for images, labels in train_batch.take(1):
    print(images, labels)
# %%  # Define the Keras TensorBoard callback.
topdirname = 'D:\Data\logs\TFcertUdemy\\05food10clsTransLearn2\\'
tbcb, logdir = tps.create_tb_callback(topdirname, 'tf_hub_src', 'effnetB0_10percent')
#%%  Build a data augmentation layer
data_augmentation = Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.RandomRotation(10) # 10 degrees
    ], name='data_augmentation')
#%% Setup input shape
input_shape = (224, 224, 3)
# Create a frozen base model (also called the backbone)
#%% build the model from tf.keras.applications (not from some URL as in 04transfer learning...)
effnetB0_base = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)
# , weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=10,
#     classifier_activation='softmax'
# 2. Freeze the base model (so the pre-learned patterns remain)
effnetB0_base.trainable = False

# 3. Create inputs into the base model
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

# 4. If using models like ResNet50V2, add this to speed up convergence, remove for EfficientNet
# x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

# 5. Pass the inputs to the effnetB0_base (note: using tf.keras.applications, EfficientNet inputs don't have to be normalized)
x = effnetB0_base(inputs)
# Check data shape after passing it to effnetB0_base
print(f"Shape after effnetB0_base: {x.shape}")

# 6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"After GlobalAveragePooling2D(): {x.shape}")

# 7. Create the output activation layer
outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

# 8. Combine the inputs with the outputs into a (new) model
effnetB0_0 = tf.keras.Model(inputs, outputs)

# 9. Compile the model
effnetB0_0.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# 10. Fit the model (we use less steps for validation so it's faster)
starttime = time.perf_counter()
effnetb0hist = effnetB0_0.fit(train_batch,
                                 epochs=5,
                                 steps_per_epoch=len(train_batch),
                                 validation_data=test_batch,
                                 # Go through less of the validation data so epochs are faster (we want faster experiments!)
                                 validation_steps=int(0.25 * len(test_batch)), 
                                 # Track our model's training logs for visualization later
                                 callbacks=[tbcb])
endtime = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
effnetB0_0.evaluate(test_batch)
# %% Prepare df's for the chart
# record time data
effnetb0 = round(endtime - starttime,2)
effnetb0df = pd.DataFrame(effnetb0hist.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetb0df.rename(columns = {'index':'epochs'}, inplace=True)
effnetb0df

# effnetb0 = effnetB0_base.summary()
stringlist2 = []
effnetB0_0.summary()
effnetB0_0.summary(print_fn=lambda x: stringlist2.append(x))
effnetB0mdlsum = "\n".join(stringlist2)
effnetB0mdlsum = effnetB0mdlsum.replace('_________________________________________________________________\n', '')
# Check layers in our base model
for layer_number, layer in enumerate(effnetB0_base.layers):
  print(layer_number, layer.name)
  
#effnetB0_base.summary() # 1280 lines? A lot, anyway, fills teh console
