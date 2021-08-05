# -*- coding: utf-8 -*-
'''
Created on Mon Jul 12 11:23:45 2021
Daniel Bourke's class '
Lessons 139 thru 148.
@author: steve

Conda Envronment:   TF25pyt39
Gpu  Support:       True
Cuda Support:       True
Tensor Flow:        2.5.0
Python version:      3.9.6.
The numpy version:   1.21.0.
The panda version:   1.3.0.
Tensorboard version  2.5.0.
Mon Jul 26 14:03:25 2021       

Also:
Conda Envronment:   Spyder
Gpu  Support:       True
Tensor Flow:        2.5.0
Python version:      3.9.6.
The numpy version:   1.21.0.
Mon Jul 26 16:22:59 2021    
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 462.75       Driver Version: 462.75       CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+

Downloads models (by the code). places them in:
    C:\\Users\steve\AppData\Local\Temp\tfhub_modules
'''
#%% Imports...
import numpy as np, matplotlib.pyplot as plt, os as os, pandas as pd, seaborn as sns
# 1 July 2021... next two statements...BEFORE any tensorflow did the trick.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time , sys, tensorflow as tf, tensorboard, sklearn.metrics, itertools, io

import random

from datetime import datetime
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorboard.plugins.hparams import api as hp

# import pydot
# import graphviz  #for graph of 
import GPUtil
gpus = GPUtil.getGPUs()
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import tensorPrepStarter as tps

# !nvidia-smi
# %%  Get the Data  commented out after completed
import zipfile
import wget # conda's python-wget

url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip'
output ='d:/data/udemy/dbourkeTFcert'
dwnldFile = wget.download(url, out=output)  # this worked! :)
destination = os.path.join(output, dwnldFile)  # this join gives two dif slashes :(

zip_ref = zipfile.ZipFile(dwnldFile, "r")
#zipfile.ZipFile.namelist('d:\\data\\udemy\\dbourkeTFcert\\10_Food_classes_10percent.zip')
zipfile.ZipInfo.filename
zip_ref.extractall(output)
zip_ref.close()

destination = 'd:\\data\\udemy\\dbourkeTFcert\\10_Food_classes_10_percent'
for dirpath, dirnames, filenames in os.walk(destination):
    print(f'There are {len(dirnames)} images and {len(filenames)} in {dirpath}')

# setup the train and test directories...
train_dir = 'd:/data/udemy/dbourkeTFcert/10_Food_Classes_10_percent/train/'
test_dir  = 'd:/data/udemy/dbourkeTFcert/10_Food_Classes_10_percent/test/'

import pathlib
data_dir = pathlib.Path(train_dir) # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print('class names are: ',class_names)

tps.view_random_image(train_dir, random.choice(class_names))

# %% Create the data loaders
# normalize 0 to 255; turn data into "Flows of batches of data"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SHAPE= (224, 224)  # HYPER PARAMETER
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen  = ImageDataGenerator(rescale=1/255.)

print("Training images:")
train_10pcbatch =\
    train_datagen.flow_from_directory(train_dir, 
                                      target_size=IMAGE_SHAPE,
                                      batch_size=BATCH_SIZE,
                                      seed=42)
print('Test images:')
test_10pcbatch = \
    test_datagen.flow_from_directory(test_dir,
                                     target_size=IMAGE_SHAPE,
                                     batch_size=BATCH_SIZE,
                                     seed=42)
    
# %% [5] # Define the Keras TensorBoard callback.
logdir="d:/data/logs/TFcertUdemy/04food10cls/" 
def create_tb_callback(dirname, expname):
    log_dir = logdir +dirname + '/' + expname+'_'  + datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
    hparams_callback = hp.KerasCallback(logdir, {'num_relu_units': 512,
                                    'dropout': 0.2})
    return tensorboard_callback

# tf.keras.callbacks.\
#     TensorBoard(log_dir=logdir, histogram_freq=0, batch_size=32, 
#                 write_graph=True, write_grads=False, write_images=False, 
#                 embeddings_freq=0, embeddings_layer_names=None, 
#                 embeddings_metadata=None, embeddings_data=None, 
#                 update_freq='epoch')
# Define the per-epoch callback. Confusion matrix
# cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# #%% Lesson 42 Create model with tensorflow hub
# # pulled this down from the tensorflowhub site...using the functionalized version from Daniel instead.
# num_classes = 10
# import tensorflow_hub as hub
# from tensorflow.keras import layers
# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
#                     trainable=False),  # Can be True, see below.
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])
# # m.build([None, expect_img_size, expect_img_size, 3])  # Batch input shape.    
# m.build([None, 224, 224, 3])  # Batch input shape.    

# %% Sources of models are:
# Resnet 50 V2 feature vector
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

# EfficientNet0 feature vector
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
# %% func: pull in and define model
import tensorflow_hub as hub
import tensorflow.keras.layers as layers
def create_model(model_url, num_classes=10, mdl_name='Steve1'):
    '''
    Takes a Tensorflow HUB URL and creates a Keras Sequential model
    Args:
        model_url (str):  A tensorflow hub feature extractin URL.
        num_classes (int): Number of output neuroons in the output layer,
            should be equal to number of target classes, default of 10.
    
    Returns:
        An uncompiled Keras Sequential  model with model_url as feature extractor
        layer and Dense outut layer with num_classes output neurons.
    '''
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer \
        = hub.KerasLayer(model_url,
                         trainable=False, # freeze already known patterns
                         name='feature_extraction_layer',
                         input_shape=IMAGE_SHAPE+(3,)) # add a dimension
  
    # Create our own model
    model = tf.keras.Sequential([
      feature_extractor_layer, # use the feature extraction layer as the base
      layers.Dense(num_classes, activation='softmax', name='output_layer') # create our own output layer      
    ], name = mdl_name)
    return model                                           
# %% Creating Resnet TF Hub model
tbcb = create_tb_callback('tf_hub_src', 'resnet50v2')
num_classes = 10
resnet_model = create_model(resnet_url, num_classes, 'resnet50v2')
resnet_model.summary()
resnet_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics='accuracy')
starttime = time.perf_counter()

resnethist = resnet_model.fit(train_10pcbatch, batch_size=BATCH_SIZE, epochs=5,
                 validation_data=test_10pcbatch, 
                 validation_steps=len(test_10pcbatch)
                 , callbacks=[tbcb])
endtime = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')

# %% Prepare df's for the chart
# record time data
resnetdur = round(endtime - starttime,2)
resnetdf = pd.DataFrame(resnethist.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
resnetdf.rename(columns = {'index':'epochs'}, inplace=True)
resnetdf

resnet = resnet_model.summary()
stringlist2 = []
resnet_model.summary(print_fn=lambda x: stringlist2.append(x))
resnetmdlsum = "\n".join(stringlist2)
resnetmdlsum = resnetmdlsum.replace('_________________________________________________________________\n', '')

# %% sHOW the chart...
supttl = 'Udemy TF Certify ZtoM 10 Food Classification Models Lecture 143'
lftTtl = '"By hand" Convolutional Model'
rhtTtl = 'Transfer Learning Resnet50v2'
augmnt = '''From "TensorFlowHub", the borrowed model is over twice
 faster, and over twice as accurate!!!'''
# ChartMnistNetworkChanges(histFood3df, resnetdf, foodmdl3sum, foodmdl3dur,
#                          resnetdur,resnetmdlsum,  augmnt, supttl, lftTtl,rhtTtl)
# pd.DataFrame(resnetdf.drop('epochs', axis=1)).plot(figsize=(10,7))

# %% Now, bring in Efficient net...compare to resnet...
tbcb = create_tb_callback('tf_hub_src', 'effnetB0')
effnet_model = create_model(efficientnet_url, num_classes, 'EfficientNetB0')
effnet_model.summary()
effnet_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics='accuracy')
starttime = time.perf_counter()

effnethist = effnet_model.fit(train_10pcbatch, batch_size=BATCH_SIZE, epochs=5,
                 validation_data=test_10pcbatch, 
                 validation_steps=len(test_10pcbatch)
                 , callbacks=[tbcb])
endtime = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
# %% Prepare df's for the chart
# record time data
effnetdur = round(endtime - starttime,2)
effnetdf = pd.DataFrame(effnethist.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
effnetdf.rename(columns = {'index':'epochs'}, inplace=True)
effnetdf

effnet = effnet_model.summary()
stringlist2 = []
effnet_model.summary(print_fn=lambda x: stringlist2.append(x))
effnetmdlsum = "\n".join(stringlist2)
effnetmdlsum = effnetmdlsum.replace('_________________________________________________________________\n', '')
# %% sHOW the chart...
supttl = 'Udemy TF Certify ZtoM 10 Food Classification Models Lecture 145'
lftTtl = 'Transfer "Resnet50v2" CNN'
rhtTtl = 'Transfer "EfficientNet" CNN'
augmnt = '''From "TensorFlowHub", the efficientNet model is even faster
 , and more accurate still!!!  AND, only using 10% of the 
 training images...which is why it is so fast.
 
 All features (weights/bias) in the feature extractor are 
 frozen!
 
 This is FEATURE EXTRACTION TRANSFER LEARNING.'''
tps.ChartMnistNetworkChanges(resnetdf,effnetdf, resnetmdlsum, resnetdur,
                         effnetdur,effnetmdlsum,  augmnt, supttl, lftTtl,rhtTtl)
