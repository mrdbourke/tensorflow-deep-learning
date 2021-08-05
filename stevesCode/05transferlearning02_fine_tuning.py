# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:06:12 2021
Most important takeaways:... 
    1. image_dataset_from_directory: Faster, simpler
    2. Functional API
@author: steve
Conda Envronment:   MLflow
Gpu  Support:       True
Cuda Support:       True
Tensor Flow:        2.4.1
Python version:      3.8.8.
The numpy version:   1.19.5.
The panda version:   1.2.4.
Tensorboard version  2.4.1.
July 26, 2021

"""
# %% Imports...
import numpy as np, matplotlib.pyplot as plt
import os, pandas as pd, seaborn as sns
# 1 July 2021... next two statements...BEFORE any tensorflow did the trick.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time , sys, tensorflow as tf, sklearn.metrics, itertools, io

import random, tensorboard

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
import graphviz  #for graphs; just python-graphviz  'graphviz' was not enough (what's the difference?) 
import GPUtil
gpus = GPUtil.getGPUs()
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)
path = 'C:/Users/steve/Documents/GitHub/DeepLearning/Udemy/tensorflow-deep-learning/stevesCode'
sys.path
sys.path.insert(0,path)
import tensorPrepStarter as tps

# !nvidia-smi

# %% get data
# url = url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip'
# zipfilepath = 'd:/data/udemy/dbourkeTFcert'
# tps.unzip_data(zipfilepath, url) 
# %%  # Define the Keras TensorBoard callback.
topdirname = 'D:\Data\logs\TFcertUdemy\\05food10clsTransLearn2\\'
tbcb, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB0', '10perc_noAug')
# %% tps.ShowImageFolder_SetTrainTestDirs
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_10_percent'
#imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_1_percent'

train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)
#%% define:  checkpoints
checkpoint_path = 'D:/Data/udemy/dbourkeTFcert/ten_percent_model_checkpoints_weights/checkpoint.ckpt' # note: remember saving directly to Colab is temporary
checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_path,
                    save_weights_only=True, # set to False to save the entire model
                    save_best_only=True, # set to True to save only the best model instead of a model every epoch 
                    save_freq="epoch", # save every epoch
                    verbose=1)

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
# below is optional...just if you want to see what the data looks like...
# for images, labels in train_batch.take(1):
#     print(images, labels)
#%% Setup input shape
input_shape = (224, 224, 3)
# Create a frozen base model (also called the backbone)
#%% 1. build the model from tf.keras.applications (not from some URL as in 04transfer learning...)
effnetB0_base = tf.keras.applications.efficientnet.\
    EfficientNetB0(include_top=False)
# effnetB0_base = tf.keras.applications.resnet50.ResNet50()
# the model trained on imagenet with 1000 classes, has 1000 outputs
# , weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=10,
#     classifier_activation='softmax'
# 2. Freeze the base model (so the pre-learned patterns remain)
effnetB0_base.trainable = False

# 3. Create inputs into the base model...what should the model expect as input?
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

# 4. If using models like ResNet50V2, add this to speed up convergence, 
#    remove for EfficientNet 
# x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

# 5. Pass the inputs to the effnetB0_base (note: using tf.keras.applications, 
     # EfficientNet inputs don't have to be normalized)
x = effnetB0_base(inputs)
# Check data shape after passing it to effnetB0_base
print(f"Shape after effnetB0_base: {x.shape}")  # (none, 7, 7, 1280)

# 6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
x = tf.keras.layers.GlobalAveragePooling2D\
    (name="global_average_pooling_layer")(x)    # (none, 1280)
    
print(f"After GlobalAveragePooling2D(): {x.shape}")

# 7. Create the output activation layer
outputs = tf.keras.layers.Dense\
    (10, activation="softmax", name="output_layer")(x)

# 8. Combine the inputs with the outputs into a (new) model
effnetB0_0 = tf.keras.Model(inputs, outputs, name='effnetb0_base')

# 9. Compile the model
effnetB0_0.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# 10. Fit the model (we use less steps for validation so it's faster)
modelstart = time.strftime('%c')
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
# %% Prepare 1st model df's for the chart
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
# Check layers in our base model ... prints 236 lines...one per layer
# for layer_number, layer in enumerate(effnetB0_base.layers):
#   print(layer_number, layer.name)
  
#effnetB0_base.summary() # 1280 lines? A lot, anyway, fills teh console
#%% Description of process...no code here
'''
Running a series of transfer learning experiments
We've seen the incredible results of transfer learning on 10% of the training data, what about 1% of the training data?

What kind of results do you think we can get using 100x less data than the original CNN models we built ourselves?

Why don't we answer that question while running the following modelling experiments:

model_1: Use feature extraction transfer learning on 1% of the training data with data augmentation.
model_2: Use feature extraction transfer learning on 10% of the training data with data augmentation.
model_3: Use fine-tuning transfer learning on 10% of the training data with data augmentation.
model_4: Use fine-tuning transfer learning on 100% of the training data with data augmentation.
'''
# %% Experiment #2...
tbcb1, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB0', '01perc_DataAug')
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_1_percent'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it
#%%  Build a data augmentation layer with picture displaying difference
data_augmentation = Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.RandomRotation(10) # 10 degrees
    ], name='data_augmentation')
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.suptitle(f'Compare two images {modelstart}')
plt.title ('before')
image = tps.view_random_image(train_dir, random.choice(train_batch.class_names))
plt.subplot(1,2,2)
plt.title('after')
plt.imshow((tf.squeeze(data_augmentation(tf.expand_dims(image, axis=0))))/255.)
plt.axis('off')
plt.show()
#%% 2. build 2nd model from tf.keras.applications (not from some URL as in 04transfer learning...)
# This shows that training on 7 images per class ... is not enough training data!
basemdl1 = tf.keras.applications.efficientnet.\
    EfficientNetB0(include_top=False)
basemdl1.trainable = False
#create 'inputs'... which start out independent
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs)
#below, we connect the 'inputs' to the effnetB0_base
x = basemdl1(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D\
    (name="global_average_pooling_layer")(x)    # (none, 1280)
outputs = tf.keras.layers.Dense\
    (10, activation="softmax", name="output_layer")(x)
# 8. Combine the inputs with the outputs into a (new) model
effnetB0_1 = tf.keras.Model(inputs, outputs, name='effnetb0_1_dataaug')
effnetB0_1.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
modelstart1 = time.strftime('%c')
starttime1 = time.perf_counter()
effnetb0_1hist = effnetB0_1.fit(train_batch,
                                 epochs=5,
                                 steps_per_epoch=len(train_batch),
                                 validation_data=test_batch,
                                 # Go through less of the validation data so epochs are faster (we want faster experiments!)
                                 validation_steps=int(0.25 * len(test_batch)), 
                                 # Track our model's training logs for visualization later
                                 callbacks=[tbcb1])
endtime1 = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
# %% Prepare 2nd model df's for the chart
# record time data
effnetb0_1 = round(endtime1 - starttime1,2)
effnetb0_1df = pd.DataFrame(effnetb0_1hist.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetb0_1df.rename(columns = {'index':'epochs'}, inplace=True)
effnetb0_1df

# effnetb0_1 = effnetb0_1_base.summary()
stringlist1 = []
effnetB0_1.summary()
effnetB0_1.summary(print_fn=lambda x: stringlist1.append(x))
effnetB0_1mdlsum = "\n".join(stringlist1)
effnetB0_1mdlsum = effnetB0_1mdlsum.replace('_________________________________________________________________\n', '')

# %% Experiment #3 (customize)...
tbcb, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB0', '10perc_DataAug')
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_10_percent'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it

#%% 2. build 3rd model from tf.keras.applications (not from some URL as in 04transfer learning...)
# This shows that training on 7 images per class ... is not enough training data!
basemdl2 = tf.keras.applications.efficientnet.\
    EfficientNetB0(include_top=False)
basemdl2.trainable = False
#create 'inputs'... which start out independent
inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs)
#below, we connect the 'augmented inputs' (via the 'x' to the effnetB0_base
x = basemdl2(x, training=False)
x = layers.GlobalAveragePooling2D\
    (name="global_average_pooling_layer")(x)    # (none, 1280)
outputs = layers.Dense\
    (10, activation="softmax", name="output_layer")(x)
# 8. Combine the inputs with the outputs into a (new) model
effnetB0_2 = tf.keras.Model(inputs, outputs, name='effnetB0_2_dataaug')

effnetB0_2.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
modelstart2 = time.strftime('%c')
starttime2 = time.perf_counter()
effnetb2hist = effnetB0_2.fit(train_batch,
                                 epochs=5,
                                 steps_per_epoch=len(train_batch),
                                 validation_data=test_batch,
                                 # Go through less of the validation data so epochs are faster (we want faster experiments!)
                                 validation_steps=int(0.25 * len(test_batch)), 
                                 # Track our model's training logs for visualization later
                                 callbacks=[tbcb, checkpoint_callback])
endtime2 = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
# %% Prepare 2nd model df's for the chart
# record time data
effnetb0_2 = round(endtime - starttime,2)
effnetb0_2df = pd.DataFrame(effnetb2hist.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetb0_2df.rename(columns = {'index':'epochs'}, inplace=True)
effnetb0_2df

# effnetb0_2 = effnetb0_2_base.summary()
stringlist1 = []
effnetB0_2.summary()
effnetB0_2.summary(print_fn=lambda x: stringlist1.append(x))
effnetB0_2mdlsum = "\n".join(stringlist1)
effnetB0_2mdlsum = effnetB0_2mdlsum.replace('_________________________________________________________________\n', '')
#%% graph differences 
supttl = 'Udemy TFCertifyZtoM 10 Food Classification TransLearn Efficient Net Lect 163'
lftTtl = 'CNN No data augmentation '
rhtTtl = 'CNN EffnetB0 with data augmentation'
augmnt = '''From "TensorFlowHub", the augmented model is less accurate!!!'''

# tps.ChartMnistNetworkChanges(dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, rmdlsumry, changes, supttl, ttl1, ttl2)
tps.ChartMnistNetworkChanges(effnetb0df, effnetb0_2df, effnetB0mdlsum, 
                             effnetb0, effnetb0_2, effnetB0_2mdlsum, 
                             augmnt, supttl, lftTtl, rhtTtl)
# %% Experiment #4 FINETUNING (customize)...
tbcb, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB0', '10perc_DataAug_fintun10')
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_10_percent'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it
checkpoint_path = 'D:/Data/udemy/dbourkeTFcert/ten_percent_finetune_checkpoints_weights/checkpoint.ckpt' # note: remember saving directly to Colab is temporary
checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_path,
                    save_weights_only=True, # set to False to save the entire model
                    save_best_only=True, # set to True to save only the best model instead of a model every epoch 
                    save_freq="epoch", # save every epoch
                    verbose=1)
    
#%% FINE TUNING
# To begin fine-tuning, we'll unfreeze the entire base model by setting its 
# trainable attribute to True. Then we'll refreeze every layer in the base 
# model except for the last 10 by looping through them and setting their 
# trainable attribute to False. Finally, we'll recompile the model.

basemdl2.trainable = True

for layer in basemdl2.layers[:-10]:
    layer.trainable = False
    
effnetB0_2.compile(loss='categorical_crossentropy',
                   optimizer=Adam(lr=0.0001), # lr is 10x lower than before for fine-tuning
                   metrics='accuracy')
for layer_number, layer in enumerate(basemdl2.layers):
    print(layer_number, layer.name, layer.trainable)

# Fine tune for another 5 epochs...
fine_tune_epochs = 5 + 5
# Refit the model...
effnetb2histft = effnetB0_2.fit(train_batch, epochs=fine_tune_epochs,
                                validation_data=test_batch,
                                initial_epoch=effnetb2hist.epoch[-1], # start from previous last epoch
                                validation_steps=int(0.25 * len(test_batch)),
                                callbacks=[tbcb, checkpoint_callback])
# %% Prepare Fine tuned model df's for the chart
# record time data
effnetb0_2dur = round(endtime - starttime,2)
effnetb0_3df = pd.DataFrame(effnetb2histft.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetb0_3df.rename(columns = {'index':'epochs'}, inplace=True)
effnetb0_3df

effnetb0_3 = effnetB0_2.summary()
stringlist1 = []
effnetB0_2.summary(print_fn=lambda x: stringlist1.append(x))
effnetB0_3mdlsum = "\n".join(stringlist1)
effnetB0_3mdlsum = effnetB0_3mdlsum.replace('_________________________________________________________________\n', '')
#%% graph differences 
supttl = 'Udemy TFCertifyZtoM 10 Food Classification TransLearn Efficient Net Lect 166'
lftTtl = 'CNN EffnetB0 with data augmentation'
rhtTtl = 'CNN EffnetB0 w/data aug and fine tuned'
augmnt = '''Fine tuning takes up right where the 1st quit
adding 6 more epochs of training at LR = .0001, 
training the last 10 layers of the base effnetB0 model'''

# tps.ChartMnistNetworkChanges(dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, rmdlsumry, changes, supttl, ttl1, ttl2)
tps.ChartMnistNetworkChanges(effnetb0_2df,effnetb0_3df, effnetB0_2mdlsum, 
                             effnetb0_2, effnetb0_2, effnetB0_3mdlsum, 
                             augmnt, supttl, lftTtl, rhtTtl)
# %% Experiment #5 FINETUNE on ALL DATA (customize)...
# we're going to have a model 'grossly' trained on 10%, finetuned on ALL data.
tbcb, logdir = tps.create_tb_callback\
    (topdirname, 'effnetB0', 'Alldata_DataAug_fintun10layers')
imagefolder = 'd:/data/udemy/dbourkeTFcert/10_food_classes_all_data'
train_dir, test_dir = tps.ShowImageFolder_SetTrainTestDirs(imagefolder)

train_batch = tf.keras.preprocessing.\
    image_dataset_from_directory(train_dir, 
                                 image_size=IMAGE_SHAPE,
                                 batch_size=BATCH_SIZE, seed = 42,
                                 label_mode='categorical') #this is the default, so i don't really need it
checkpoint_path = 'D:/Data/udemy/dbourkeTFcert/Alldata_finetune_checkpoints_weights/checkpoint.ckpt' # note: remember saving directly to Colab is temporary
checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_path,
                    save_weights_only=True, # set to False to save the entire model
                    save_best_only=True, # set to True to save only the best model instead of a model every epoch 
                    save_freq="epoch", # save every epoch
                    verbose=1)

#%% EXP 5: restore 'model 2'
effnetB0_2.load_weights(checkpoint_path)  # this is a change to the model... therefore will need to recompile.
effnetB0_2.evaluate(test_batch)
for layer in effnetB0_2.layers:
    print(layer.name, layer.trainable)
for layer in basemdl2.layers:
    if layer.trainable == True:  print (layer.name)

effnetB0_2full = effnetB0_2
#%% Compile & fit
effnetB0_2full.compile(loss='categorical_crossentropy', 
                       optimizer=Adam(lr=0.0001),
                       metrics='accuracy')
effnetb2histftAll = effnetB0_2full.fit(
    train_batch, epochs=fine_tune_epochs,
    validation_data=test_batch,
    initial_epoch=effnetb2hist.epoch[-1], # start from previous last epoch
    validation_steps=int(0.25 * len(test_batch)),
    callbacks=[tbcb, checkpoint_callback])
