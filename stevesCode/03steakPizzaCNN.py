# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:29:32 2021

@author: steve
"""

#%% Imports...
import time , sys, tensorflow as tf, tensorboard, sklearn.metrics, itertools, io
from datetime import datetime
from tensorflow import keras
import numpy as np, matplotlib.pyplot as plt, os, pandas as pd, seaborn as sns
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import pydot
import graphviz  #for graph of 
import GPUtil
gpus = GPUtil.getGPUs()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import logging
tf.get_logger().setLevel(logging.ERROR)

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

#%%  ChartMnistNetworkChanges
def ChartMnistNetworkChanges(dfLeft, dfRight, mdlsummary, dfLtime, dfRtime, 
                             rmdlsumry, changes):
    modelstart = time.strftime('%c')
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.suptitle(f'Udemy TensorFlowCertZtoM COMPARE CNN to ANN on PizzaSteak pics {modelstart}')
    plt.title (f'1st CNN Model {dfLtime:.2f} sec ')
    sns.lineplot(x='epochs', y='value', hue='variable', data = pd.melt(dfLeft,['epochs']))
    plt.text(0., .5,# transform=trans1,
              s=mdlsummary,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='pink', alpha=0.5))
    plt.legend(loc= 'lower left')
    plt.ylim(.3,.95)
    plt.xlabel('epochs    \GitHub\DeepLearning')

    plt.subplot(1,2,2)
    plt.text(-.3, .45,# transform=trans1,
              s=changes,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='aqua', alpha=0.5))
    plt.text(0., .62,# transform=trans1,
              s=rmdlsumry,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='pink', alpha=0.5))
    plt.title (f'Mnist NN design on PizzaSteak {dfRtime:.2f} sec ')
    sns.lineplot(x='epochs', y='value', hue='variable', data = pd.melt(dfRight,['epochs']))
    plt.ylim(.3,.95)
    plt.xlabel('03steakPizzaCNN.py    epochs')
    plt.show();
#%%
#ChartMnistNetworkChanges(df, df1, short_model_summary, duration_n, duration)
#%% Prepare the 'dataflow'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Setup the train and test directories
train_dir = "d:/data/udemy/dbourkeTFcert/pizza_steak/train/"
test_dir  = "d:/data/udemy/dbourkeTFcert/pizza_steak/test/"

# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

#%% Create a CNN model (same as Tiny VGG - https://poloclub.github.io/cnn-explainer/)
starttime = time.perf_counter()
# Set the seed
tf.random.set_seed(42)
pzaStk1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=10, 
                         kernel_size=3, # can also be (3, 3)
                         activation="relu", 
                         input_shape=(224, 224, 3)), # first layer specifies input shape (height, width, colour channels)
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                            padding="valid"), # padding can also be 'same'
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid") # binary activation output
], name='PizzaStake1')

# Compile the model
pzaStk1.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit the model
pzaHist1 = pzaStk1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
endtime = time.perf_counter()
duration = round(endtime - starttime,2)
df = pd.DataFrame(pzaHist1.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
df.rename(columns = {'index':'epochs'}, inplace=True)
df
#%% Apply the OLDER, DENSE ANN  (no convolutions) then compare
changes = '''
# step 7 drop down to a single relu layer       ... acc: .87 time: 44.8 sec
Switch to steak & pizza pics...
'''
starttime = time.perf_counter()
tf.random.set_seed(42)
mnist4 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, 'softmax')
    ],name = 'mnist4_128nodes')
mnist4.compile(optimizer=tf.keras.optimizers.Adam(lr=.002),
               loss=tf.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy']
               )
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 **(epoch/10))
historyM4 = mnist4.fit(train_data, epochs=10,
                       validation_data=valid_data,
                       validation_steps=len(valid_data),
                       #callbacks=[lr_scheduler], 
                       verbose=1)
endtime = time.perf_counter()
duration_n = round(endtime - starttime,2)
df1 = pd.DataFrame(historyM4.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
df1.rename(columns = {'index':'epochs'}, inplace=True)
df1
stringlist = []
mnist4.summary(print_fn=lambda x: stringlist.append(x))
mnist4summary = "\n".join(stringlist)

#%% Document the model...
mdltxt = pzaStk1.summary()
stringlist = []
pzaStk1.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
print(short_model_summary)

# pzaStk1.evaluate(test_data_n, test_labels)
changes = '''You can think of trainable parameters as *patterns a model can learn from
 data*. Intuitiely, you might think more is better. And in some cases it is. 
 But in this case, the difference here is in the two different styles of model we're 
 using. Where a series of dense layers have a number of different learnable parameters 
 connected to each other and hence a higher number of possible learnable patterns, 
 **a convolutional neural network seeks to sort out and learn the most important 
 patterns in an image**. So even though there are less learnable parameters in 
 our convolutional neural network, these are often more helpful in decphering 
 between different **features** in an image.'''
ChartMnistNetworkChanges(df, df1, short_model_summary, duration_n, duration, mnist4summary, changes)
model_1.summary()
