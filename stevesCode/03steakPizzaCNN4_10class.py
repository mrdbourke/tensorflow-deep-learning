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

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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

#%% func: ChartMnistNetworkChanges dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, rmdlsumry, changes
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
#%% func: view_random_image 
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
#%%  Get the Data
'''
import zipfile
import wget
# https://www.pair.com/support/kb/paircloud-downloading-files-with-wget/
# Download zip file of pizza_steak images
# # !wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip 
url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip'
output ='d:\\data\\udemy\\dbourkeTFcert'

'''
output ='d:/data/udemy/dbourkeTFcert'
#dwnldFile = wget.download(url, out=output)  # this worked! :)
dwnldFile = '10_Food_Classes_All_Data'
destination = os.path.join(output, dwnldFile)  # this join gives two dif slashes :(
source = 'C:/Users/steve/Downloads/10_Food_classes_all_data.zip'
# Unzip the downloaded file
'''
zip_ref = zipfile.ZipFile(source, "r")
#zip_ref = zipfile.ZipFile(destination, "r")
zipfile.ZipFile.namelist('d:\\data\\udemy\\dbourkeTFcert\\pizza_steak.zip')
zipfile.ZipInfo.filename
zip_ref.extractall(output)
zip_ref.close()
'''
for dirpath, dirnames, filenames in os.walk(destination):
    print(f'There are {len(dirnames)} images and {len(filenames)} in {dirpath}')
    
# setup the train and test directories...
train_dir = 'd:/data/udemy/dbourkeTFcert/10_Food_Classes_All_Data/train/'
test_dir  = 'd:/data/udemy/dbourkeTFcert/10_Food_Classes_All_Data/test/'

import pathlib
data_dir = pathlib.Path(train_dir) # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print('class names are: ',class_names)

view_random_image(train_dir, random.choice(class_names))
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
# train_dir = "d:/data/udemy/dbourkeTFcert/pizza_steak/train/"
# test_dir  = "d:/data/udemy/dbourkeTFcert/pizza_steak/test/"

# Import data from directories and turn it into batches
train_batch = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="categorical", # type of problem we're working on
                                               #shuffle=False, #default is True
                                               seed=42)
 
valid_batch = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="categorical",
                                               seed=42)

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

#%%  # Creates a file writer for the log directory.
imagedir = logdir+'/imgs'
file_writer = tf.summary.create_file_writer(imagedir)
image_batch, labels = train_batch.next()

with file_writer.as_default():
  # Don't forget to reshape.
  images = np.reshape(image_batch[0:32], (-1, 224, 224, 3))
  tf.summary.image("25 training data examples", images, max_outputs=32, step=0)

test_summary_writer = tf.summary.create_file_writer(logdir)
with test_summary_writer.as_default():
    tf.summary.scalar('loss', 0.345, step=1)
    tf.summary.scalar('loss', 0.234, step=2)
    tf.summary.scalar('loss', 0.123, step=3)     
#%% Define the model, compile, fit
foodmdl = Sequential([
    Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation='relu', name='2nd'),
    tf.keras.layers.MaxPool2D( name='1stMax2D'),
    Conv2D(10, 3, activation='relu', name='3rd'), 
    Conv2D(10, 3, activation='relu', name='4th'),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation='softmax', name='1stDense')], name='10ClassFood_mdl')
foodmdl.compile(optimizer=Adam(), loss='categorical_crossentropy',
                metrics='accuracy')
starttime = time.perf_counter()

# model fitting returns: 'History' object (python.keras.callbacks.History)
histFood = foodmdl.fit(train_batch, batch_size=batchsize, epochs=5,
                       steps_per_epoch=len(train_batch),
                       callbacks=[tensorboard_callback],
                       validation_data=valid_batch, 
                       validation_steps=len(valid_batch))
endtime = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')

#%% show the model (which model, exactly?? need to pass in the BEST!)
# record time data
foodmdldur = round(endtime - starttime,2)
histFooddf = pd.DataFrame(histFood.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
histFooddf.rename(columns = {'index':'epochs'}, inplace=True)
histFooddf
import json
type(foodmdl.get_config().items())
stf = foodmdl.get_config().items()
list(stf)[1][1]
len(list(stf)[1][1])
type(list(stf)[1][1][4])
list(stf)[1][1][4].get('name')
str(list(stf)[1][1])
type(json.dumps(list(stf)[1][1])) # str!!!
len(stf)

list(stf)[1][1][2]['config'].get('kernel_size')  ##Sigmoid
list(stf)[1][1][4]['config'].get('activation')  ##Sigmoid
list(stf)[1][1][4]['config'].get('units')  ## 1

mdltxt = foodmdl.summary()
stringlist = []
foodmdl.summary(print_fn=lambda x: stringlist.append(x))
foodmdlsum = "\n".join(stringlist)
foodmdlsum = foodmdlsum.replace('_________________________________________________________________\n', '')
#%% Preprocess data (augmentation)
train_datagen_aug = ImageDataGenerator(rescale=1./255, rotation_range=0.2,
                                   shear_range=0.2, zoom_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.3,
                                   horizontal_flip=True)
# Import data from directories and turn it into batches
train_batch_aug = train_datagen_aug.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="categorical", # type of problem we're working on
                                               #shuffle=False, #default is True
                                               seed=42)
#%% Define the 2nd model, compile, fit
# below is NOT as good as prior...
# below we went to 3 conv layers, and 8 filters per layer.  NOT better
foodmdl2 = Sequential([
    Conv2D(8, 3, activation='relu', input_shape=(224, 224, 3), name='8filt'),
    tf.keras.layers.MaxPool2D( name='1st'),
    Conv2D(8, 3, activation='relu', name='8filters'), 
    MaxPool2D(name='2nd'),
    Conv2D(8, 3, activation='relu', name='8filter'), 
    MaxPool2D(name='3rd'),
    Flatten(),
    Dense(10, activation='softmax', name='1stDense')], name='10ClassFood_mdl2fewerFilters')
foodmdl2.compile(optimizer=Adam(), loss='categorical_crossentropy',
                metrics='accuracy')
starttime = time.perf_counter()

# model fitting returns: 'History' object (python.keras.callbacks.History)
histFood2 = foodmdl2.fit(train_batch_aug, batch_size=batchsize, epochs=5,
                       steps_per_epoch=len(train_batch_aug),
                       callbacks=[tensorboard_callback],
                       validation_data=valid_batch, 
                       validation_steps=len(valid_batch))
endtime = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% Prepare df's for the chart
# record time data
foodmdl2dur = round(endtime - starttime,2)
histFood2df = pd.DataFrame(histFood2.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
histFood2df.rename(columns = {'index':'epochs'}, inplace=True)
histFood2df

mdltxt2 = foodmdl2.summary()
stringlist2 = []
foodmdl2.summary(print_fn=lambda x: stringlist2.append(x))
foodmdl2sum = "\n".join(stringlist2)
foodmdl2sum = foodmdl2sum.replace('_________________________________________________________________\n', '')

#%% sHOW the chart...
supttl = 'Udemy TF Certify ZtoM 10 Food Classification Models Lecture 130'
lftTtl = 'BaseLine Convolutional Model'
rhtTtl = 'CNN with Augmented Data'
augmnt = '''Data augmentation: rotation_range=0.2, shear_range=0.2, 
zoom_range=0.2,width_shift_range=0.2, 
height_shift_range=0.3, horizontal_flip=True'''
ChartMnistNetworkChanges(histFooddf, histFood2df, foodmdlsum, foodmdldur,
                         foodmdl2dur,foodmdl2sum,  augmnt, supttl, lftTtl,rhtTtl)
pd.DataFrame(histFood2df.drop('epochs', axis=1)).plot(figsize=(10,7))

#%% func: load_and_prep_image
# Create a function to import an image and resize it to be able to be used with our modle
def load_and_prep_image(filename, img_shape=224):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img
#%% func: pred_and_plot
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

    # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);
  
#%% Get some test images...
# -q is for "quiet"
import wget
# https://www.pair.com/support/kb/paircloud-downloading-files-with-wget/
url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip'
url = 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-pizza-dad.jpeg'
url = 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg'
url = 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-hamburger.jpeg'
url = 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-sushi.jpeg'
output ='d:\\data\\udemy\\dbourkeTFcert'
output ='d:/data/udemy/dbourkeTFcert'
dwnldFile = wget.download(url, out=output)  # this worked! :)
filename = output + '/03-hamburger.jpeg'
filename = output + '/03-sushi.jpeg'
filename = output + '/03-pizza-dad.jpeg'
filename = output + '/03-steak.jpeg'
hmb = load_and_prep_image(filename)
hmb = tf.expand_dims(hmb, axis=0)
hmb.shape
pred = foodmdl2(hmb)
pred_and_plot(foodmdl2, filename, class_names)

#%% # Get data batch samples
import random
# images, labels = train_batch.next()
# augmented_images, augmented_labels = train_batch_aug.next() # Note: labels aren't augmented, they stay the same
images, labels = train_batch.next()
augmented_images, augmented_labels = train_batch_aug.next() # Note: labels aren't augmented, they stay the same
# Show original image and augmented image
random_number = random.randint(0, 31) # we're making batches of size 32, so we'll get a random instance
plt.imshow(images[random_number])
plt.title(f"Original image {random_number}")
plt.axis(False)
plt.figure()
plt.imshow(augmented_images[random_number])
plt.title(f"Augmented image  {random_number}")
plt.axis(False);

#%% CLONE earlier model!
foodmdl3 = tf.keras.models.clone_model(foodmdl)
foodmdl3.compile(optimizer=Adam(), loss='categorical_crossentropy',
                metrics='accuracy')
starttime = time.perf_counter()

# model fitting returns: 'History' object (python.keras.callbacks.History)
histFood3 = foodmdl3.fit(train_batch_aug, batch_size=batchsize, epochs=5,
                       steps_per_epoch=len(train_batch_aug),
                       callbacks=[tensorboard_callback],
                       validation_data=valid_batch, 
                       validation_steps=len(valid_batch))
endtime = time.perf_counter()
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% Prepare df's for the chart
# record time data
foodmdl3dur = round(endtime - starttime,2)
histFood3df = pd.DataFrame(histFood3.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
histFood3df.rename(columns = {'index':'epochs'}, inplace=True)
histFood3df

mdltxt3 = foodmdl3.summary()
stringlist3 = []
foodmdl3.summary(print_fn=lambda x: stringlist3.append(x))
foodmdl3sum = "\n".join(stringlist3)
foodmdl3sum = foodmdl3sum.replace('_________________________________________________________________\n', '')

#%% sHOW the chart...
supttl = 'Udemy TF Certify ZtoM 10 Food Classification Models Lecture 135'
lftTtl = 'Simple Convolutional Model-Data Augmented'
rhtTtl = 'Base CNN with Augmented Data'
augmnt = '''Data augmentation: rotation_range=0.2, shear_range=0.2, 
zoom_range=0.2,width_shift_range=0.2, 
height_shift_range=0.3, horizontal_flip=True'''
ChartMnistNetworkChanges(histFood2df, histFood3df, foodmdl2sum, foodmdl2dur,
                         foodmdl3dur,foodmdl3sum,  augmnt, supttl, lftTtl,rhtTtl)
pd.DataFrame(histFood3df.drop('epochs', axis=1)).plot(figsize=(10,7))

