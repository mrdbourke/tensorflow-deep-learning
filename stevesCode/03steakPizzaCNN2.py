# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 20:50:24 2021

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
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
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
#%% view_random_image
def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = np.random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img
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
              fontsize=7, bbox=dict(facecolor='pink', alpha=0.5))
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
              fontsize=7, bbox=dict(facecolor='pink', alpha=0.5))
    plt.title (f'{ttl2} {dfRtime:.2f} sec ')
    sns.lineplot(x='epochs', y='value', hue='variable', data = pd.melt(dfRight,['epochs']))
    plt.ylim(.3,1)
    plt.xlabel('03steakPizzaCNN.py    epochs')
    plt.legend(loc= 'lower left')
    plt.show();
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
                                               seed=42)

valid_batch = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)
#%%  define, compile, train/fit the model...
pzastk1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation= 'relu'),
    tf.keras.layers.MaxPool2D(4),
    tf.keras.layers.Conv2D(10,3, activation='relu'),
    tf.keras.layers.Conv2D(10,3, activation='relu'),
    tf.keras.layers.MaxPool2D(4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='PizzaStake1')
pzastk1.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics='accuracy')
pzastk1.fit(train_batch, epochs=5, batch_size=32, validation_data=valid_batch,
            validation_steps=len(valid_batch))
#%% view what's created..
pzastk1.summary()
image_batch, labels = train_batch.next()
len(image_batch), len(labels)
# first two... Image values are 'red', 'green', 'blue' (a tensor of (3) for each pixel)
image_batch[1].shape  # (224, 224, 3)  tensors: the "3" is the lowest level (representing a pixel...R, G, B)
image_batch[:2]  #
# img = view_random_image(target_dir="d:/data/udemy/dbourkeTFcert/pizza_steak/train/",
#                         target_class="pizza")

img = plt.imshow(image_batch[0])
img = plt.imshow(image_batch[2])
type(img)
#plt.imshow(img)


#%%  # Define the Keras TensorBoard callback.
logdir="d:/data/logs/TFcertUdemy/03PizzaStk/" + datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 

# Define the per-epoch callback. Confusion matrix
# cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

#%%  # Creates a file writer for the log directory.
imagedir = logdir+'/imgs'
file_writer = tf.summary.create_file_writer(imagedir)

with file_writer.as_default():
  # Don't forget to reshape.
  images = np.reshape(image_batch[0:25], (-1, 224, 224, 3))
  tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

test_summary_writer = tf.summary.create_file_writer(logdir)
with test_summary_writer.as_default():
    tf.summary.scalar('loss', 0.345, step=1)
    tf.summary.scalar('loss', 0.234, step=2)
    tf.summary.scalar('loss', 0.123, step=3)     
#%% Create model 2
starttime = time.perf_counter()
# Set the seed
tf.random.set_seed(42)

pzastk2 = Sequential([
    Conv2D(filters=10, kernel_size=3, activation='relu', strides=1,
           padding='Valid', input_shape=(224,224,3)),
    Conv2D(10, 3, activation='relu'),
    Conv2D(10, 3, activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')], name = 'pizzastk2')\
# Compile the model
pzastk2.compile(loss='binary_crossentropy', optimizer=Adam(), metrics='accuracy')
# Train the model
pzaHist2 =  pzastk2.fit(train_batch, epochs=5, batch_size=32, validation_data=valid_batch,
            validation_steps=len(valid_batch), callbacks=[tensorboard_callback]) #, cm_callback])
# record time data
endtime = time.perf_counter()
pzastk2dur = round(endtime - starttime,2)
df = pd.DataFrame(pzaHist2.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
df.rename(columns = {'index':'epochs'}, inplace=True)
df
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% document the model
mdltxt = pzastk2.summary()
stringlist = []
pzastk2.summary(print_fn=lambda x: stringlist.append(x))
pzastk2mdlsum = "\n".join(stringlist)
       
#%% 3rd model
# define the model
starttime = time.perf_counter()
pzastk3 = Sequential([
    Conv2D(10, 3, activation='relu', input_shape=(224,224,3), name='conv2d_1st',
           padding='same'),
    Conv2D(10, 3, activation='relu', name='conv2d_2nd'),
    tf.keras.layers.MaxPool2D(pool_size=2, # this is REGULARIZATION!
                            padding="valid"), # padding can also be 'same'
    Conv2D(10, 3, activation='relu', name='conv2d_3rd',
           padding='same'),
    Conv2D(10, 3, activation='relu', name='conv2d_4th'),
    tf.keras.layers.MaxPool2D(pool_size=2, # GETS the most IMPORTANT FEATURE (of the 4 pixel sets)
                            padding="valid"), # padding can also be 'same'
    Flatten( name='flatten_1st'),
    Dense(1, activation='sigmoid')
           ], name = 'pizzastk3')
# compile the model
pzastk3.compile(loss='binary_crossentropy', optimizer=Adam(), metrics='accuracy')
# fit the model
pzaHist3 = pzastk3.fit(train_batch, epochs=5, batch_size=32, validation_data = valid_batch,
                       validation_steps=len(valid_batch), callbacks=[tensorboard_callback])
# record time data
endtime = time.perf_counter()
pzastk3dur = round(endtime - starttime,2)
df3 = pd.DataFrame(pzaHist3.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
df3.rename(columns = {'index':'epochs'}, inplace=True)
df3
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
#%% document the model
import json
type(pzastk3.get_config().items())
stf = pzastk3.get_config().items()
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

mdltxt = pzastk3.summary()
stringlist = []
pzastk3.summary(print_fn=lambda x: stringlist.append(x))
pzastk3mdlsum = "\n".join(stringlist)
#%%
ChartMnistNetworkChanges(df, df3, pzastk2mdlsum, pzastk2dur, pzastk3dur,
                         pzastk3mdlsum, 'Changes', 
                         'Udemy TensorFlowCertZtoM COMPARE CNNs on PizzaSteak pics',
                         '1st CNN Model',
                         '2nd CNN Model PizzaSteak'
                         )

#%%
# from tensorflow.keras import backend as K
# K.clear_session()

# from numba import cuda
# cuda.select_device(0)
# cuda.close()
# #%%
# json_data = [] # your list with json objects (dicts)

# for item in stf:
#     for data_item in item['data']:
#         print (data_item['name'], data_item['value'])
        























