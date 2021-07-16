# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:37:04 2021

@author: steve
"""

"""
Created on Mon Jul 12 11:23:45 2021

@author: steve
"""
import numpy as np, matplotlib.pyplot as plt, os, pandas as pd, seaborn as sns
# 1 July 2021... next two statements...BEFORE any tensorflow did the trick.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time , sys, tensorflow as tf, tensorboard, sklearn.metrics, itertools, io
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorboard.plugins.hparams import api as hp
import pydot
import graphviz  #for graph of 
import GPUtil
gpus = GPUtil.getGPUs()
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# %% Get GPU status
from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())  # this puts out a lot of lines (Gibberish?)
print('Conda Envronment:  ', os.environ['CONDA_DEFAULT_ENV'])
print(f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}')
print(f'Cuda Support:       {tf.test.is_built_with_cuda()}')
print(f'Tensor Flow:        {tf.version.VERSION}')
pver = str(format(sys.version_info.major) +'.'+ format(sys.version_info.minor)+'.'+ format(sys.version_info.micro))
print('Python version:      {}.'.format(pver)) 
print('The numpy version:   {}.'.format(np.__version__))
print('The panda version:   {}.'.format(pd.__version__))
print('Tensorboard version  {}.'.format(tensorboard.__version__))
# additional imports

condaenv = os.environ['CONDA_DEFAULT_ENV']

modelstart = time.strftime('%c')

# %% func: ChartMnistNetworkChanges dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, rmdlsumry, changes
def ChartMnistNetworkChanges(dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, 
                             rmdlsumry, changes, supttl, ttl1, ttl2):
    modelstart = time.strftime('%c')
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.suptitle(f'{supttl} {modelstart}')
    plt.title (f'{ttl1} {dfLtime:.2f} sec ')
    sns.lineplot(x='epochs', y='value', hue='variable', data = pd.melt(dfLeft,['epochs']))
    plt.text(0., .5,# transform=trans1,
              s=mdlsummary,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='moccasin', alpha=0.5))
    plt.legend(loc= 'lower left')
    plt.ylim(.3,1)
    plt.xlabel('epochs    \GitHub\DeepLearning')

    plt.subplot(1,2,2)
    plt.text(.0, .95,# transform=trans1,
              s=changes,
              wrap=True, ha='left', va='top', fontname='Consolas',
              fontsize=9, bbox=dict(facecolor='aqua', alpha=0.5))
    plt.text(0., .5,# transform=trans1,
              s=rmdlsumry,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.title (f'{ttl2} {dfRtime:.2f} sec ')
    sns.lineplot(x='epochs', y='value', hue='variable', data = pd.melt(dfRight,['epochs']))
    plt.ylim(.3,1)
    plt.xlabel('04transferLearning01FeatureExtraction.py    epochs')
    plt.legend(loc= 'lower left')
    plt.show();
# %% func: view_random_image 
import matplotlib.image as mpimg
import os
import random
import tensorflow as tf
def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(f'{target_class}   {img.shape}')
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img
# %%  Get the Data  commented out after completed
import zipfile
import wget
'''
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
'''    
# setup the train and test directories...
train_dir = 'd:/data/udemy/dbourkeTFcert/10_Food_Classes_10_percent/train/'
test_dir  = 'd:/data/udemy/dbourkeTFcert/10_Food_Classes_10_percent/test/'

import pathlib
data_dir = pathlib.Path(train_dir) # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print('class names are: ',class_names)

view_random_image(train_dir, random.choice(class_names))

# %% Create the data loaders
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
# %%  # Define the Keras TensorBoard callback.
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
# %% Create model with tensorflow hub
# pulled this down from the tensorflowhub site...using the functionalized version from Daniel instead.
# num_classes = 10
# import tensorflow_hub as hub
# from tensorflow.keras import layers
# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
#                    trainable=False),  # Can be True, see below.
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
