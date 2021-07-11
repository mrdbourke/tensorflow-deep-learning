# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:49:46 2021

@author: steve
"""

#%% Imports...
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


#%% Get GPU status
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

#%%  ChartMnistNetworkChanges dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, rmdlsumry, changes
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
    plt.text(-.3, .45,# transform=trans1,
              s=changes,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='aqua', alpha=0.5))
    plt.text(0., .5,# transform=trans1,
              s=rmdlsumry,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.title (f'{ttl2} {dfRtime:.2f} sec ')
    sns.lineplot(x='epochs', y='value', hue='variable', data = pd.melt(dfRight,['epochs']))
    plt.ylim(.3,1)
    plt.xlabel('03steakPizzaCNN.py    epochs')
    plt.legend(loc= 'lower left')
    plt.show();
#%%  Read aan image...
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
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img
#%%  Get the Data
import zipfile
import wget
# https://www.pair.com/support/kb/paircloud-downloading-files-with-wget/
# Download zip file of pizza_steak images
# # !wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip 
url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip'
output ='d:\\data\\udemy\\dbourkeTFcert'


output ='d:/data/udemy/dbourkeTFcert'
#dwnldFile = wget.download(url, out=output)  # this worked! :)
dwnldFile = '10_Food_Classes_All_Data'
destination = os.path.join(output, dwnldFile)  # this join gives two dif slashes :(
source = 'C:/Users/steve/Downloads/10_Food_classes_all_data.zip'
# Unzip the downloaded file
zip_ref = zipfile.ZipFile(source, "r")
#zip_ref = zipfile.ZipFile(destination, "r")
zipfile.ZipFile.namelist('d:\\data\\udemy\\dbourkeTFcert\\pizza_steak.zip')
zipfile.ZipInfo.filename
zip_ref.extractall(output)
zip_ref.close()

for dirpath, dirnames, filenames in os.walk(destination):
    print(f'There are {len(dirnames)} images and {len(filenames)} in {dirpath}')
    
# setup the train and test directories...
train_dir = 'd:/data/udemy/dbourkeTFcert/10_Food_Classes_All_Data/train/'
test_dir  = 'd:/data/udemy/dbourkeTFcert/10_Food_Classes_All_Data/test/'

import pathlib
data_dir = pathlib.Path(train_dir) # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)

view_random_image(train_dir, 'hamburger')
#%%  # Define the Keras TensorBoard callback.
logdir="d:/data/logs/TFcertUdemy/03food10cls/" + datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
hparams_callback = hp.KerasCallback(logdir, {'num_relu_units': 512,
                                    'dropout': 0.2})

tf.keras.callbacks.\
    TensorBoard(log_dir=logdir, histogram_freq=0, batch_size=32, 
                write_graph=True, write_grads=False, write_images=False, 
                embeddings_freq=0, embeddings_layer_names=None, 
                embeddings_metadata=None, embeddings_data=None, 
                update_freq='epoch')
# Define the per-epoch callback. Confusion matrix
# cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
#%% define data source & style,  NOT AUGMENTED!
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
starttime = time.perf_counter()
# Set the seed
tf.random.set_seed(42)
# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

'''Due to how our directories are structured, the classes get inferred by the 
    subdirectory names in train_dir and test_dir.
The target_size parameter defines the input size of our images in 
    (height,  width) format.
The class_mode value of 'binary' defines our classification problem type. If 
    we had more than two classes, we would use 'categorical'.'''

# Setup the train and test directories
train_dir = "d:/data/udemy/dbourkeTFcert/pizza_steak/train/"
test_dir  = "d:/data/udemy/dbourkeTFcert/pizza_steak/test/"

# Import data from directories and turn it into batches
train_batch = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               #shuffle=False, #default is True
                                               seed=42)
 
valid_batch = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

#%% View an image
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img
#%% Get the data...
import zipfile
import wget
# Download zip file of 10_food_classes images
# See how this data was created - https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/image_data_modification.ipynb

# # !wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip 
url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip'
output ='d:\\data\\udemy\\dbourkeTFcert'
# dwnldFile = wget.download(url, out=output)  # this worked! :)
# destination = os.path.join(output, dwnldFile)  # this join gives two dif slashes :(

# # Unzip the downloaded file
# zip_ref = zipfile.ZipFile(output+"\\10_food_classes_all_data.zip", "r")
# zip_ref.extractall(output+'\\10_food_classes_all_data')
# zip_ref.close()

import os

# Walk through 10_food_classes directory and list number of files
for dirpath, dirnames, filenames in os.walk(output+"/10_food_classes_all_data"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
#%% Visualize the data

train_dir = "D:/Data/udemy/dbourkeTFcert/10_food_classes_all_data/train" #"/*.*"
test_dir  = "d:/data/udemy/dbourkeTFcert/10_food_classes_all_data/test/"
# Get the class names for our multi-class dataset
import pathlib, glob
import numpy as np
data_dir = pathlib.Path(train_dir)
# class_names = 
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)

import random
img = view_random_image(target_dir=train_dir+'/',
                        target_class=random.choice(class_names)) # get a random class name
#%% Preprocess the data
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
starttime = time.perf_counter()
# Set the seed
tf.random.set_seed(42)
batchsize = 32
targetsize = (224, 224)
# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Setup the train and test directories
train_dir = 'D:/Data/udemy/dbourkeTFcert/10_food_classes_all_data/train/'
test_dir  = "d:/data/udemy/dbourkeTFcert/10_food_classes_all_data/test/"

train_batch = train_datagen.flow_from_directory(train_dir, target_size=targetsize, 
                                  class_mode='categorical', 
                                  batch_size=batchsize, seed=42)
valid_batch = valid_datagen.flow_from_directory(test_dir, target_size=targetsize,
                                                class_mode='categorical',
                                                batch_size=batchsize, seed=42)

#%% Define the model, compile, fit
foodmdl = Sequential([
    Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    Conv2D(10, 3, activation='relu'), 
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation='softmax')    ])
foodmdl.compile(optimizer=Adam(), loss='categorical_crossentropy',
                metrics='accuracy')
histFood = foodmdl.fit(train_batch, batch_size=batchsize, epochs=1,
                       steps_per_epoch=len(train_batch),
                       callbacks=[tensorboard_callback],
                       validation_data=valid_batch, 
                       validation_steps=len(valid_batch))