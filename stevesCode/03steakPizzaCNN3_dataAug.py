# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 19:41:18 2021

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
                                               #seed=42, 
                                               shuffle=True)

valid_batch = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary") #,
                                               #seed=42)

#%% define data source & style,  AUGMENTED!
from tensorflow.keras.preprocessing.image import ImageDataGenerator
starttime = time.perf_counter()
# Set the seed
tf.random.set_seed(42)
# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen_aug = ImageDataGenerator(rescale=1./255, rotation_range=0.2,
                                   shear_range=0.2, zoom_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.3,
                                   horizontal_flip=True)
# valid_datagen_aug = ImageDataGenerator(rescale=1./255, rotation_range=0.2,
#                                    shear_range=0.2, zoom_range=0.2,
#                                    width_shift_range=0.2, height_shift_range=0.3,
#                                    horizontal_flip=True)


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
train_batch_aug = train_datagen_aug.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               #seed=42, 
                                               shuffle=True)

#%%  # Define the Keras TensorBoard callback.
logdir="d:/data/logs/TFcertUdemy/03PizzaStk/" + datetime.now().strftime("%Y%m%d_%H%M%S")
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

#%%  # Creates a file writer for the log directory.
imagedir = logdir+'/imgs'
file_writer = tf.summary.create_file_writer(imagedir)
image_batch, labels = train_batch.next()

with file_writer.as_default():
  # Don't forget to reshape.
  images = np.reshape(image_batch[0:25], (-1, 224, 224, 3))
  tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

test_summary_writer = tf.summary.create_file_writer(logdir)
with test_summary_writer.as_default():
    tf.summary.scalar('loss', 0.345, step=1)
    tf.summary.scalar('loss', 0.234, step=2)
    tf.summary.scalar('loss', 0.123, step=3)     
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
pzastk3mdlsum = pzastk3mdlsum.replace('_________________________________________________________________\n', '')

#%% hparameter design  NOT USED... good for DNN, not CNN.  Need CNN example.
# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer(logdir + '/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

#%% 4th model with data augmentation
# define the model
starttime = time.perf_counter()
pzastk4 = Sequential([
    Conv2D(10, 3, activation='relu', input_shape=(224,224,3), name='1st'),
    Conv2D(10, 3, activation='relu', name='2nd', padding='valid'),
    # MaxPool2D(name='pool0' ), # this is REGULARIZATION! padding can also be 'same'
    # Conv2D(10, 3, activation='relu', name='3rd', padding='valid'),
    # Conv2D(10, 3, activation='relu', name='4th'),
#    MaxPool2D(name='pool1'),# GETS the most IMPORTANT FEATURE (of the 4 pixel sets)
    # Conv2D(10, 3, activation='relu', name='5th'),
    # Conv2D(10, 3, activation='relu', name='6th'),
    MaxPool2D(name='pool2'),# GETS the most IMPORTANT FEATURE (of the 4 pixel sets)
    Flatten( name='flatten_1st'),
    Dense(1, activation='sigmoid', name='1stDense')
           ], name = 'pizzastk4')
# compile the model
pzastk4.compile(loss='binary_crossentropy', optimizer=Adam(lr=.001), metrics='accuracy')
# fit the model
pzaHist4 = pzastk4.fit(train_batch_aug, epochs=5, batch_size=32, 
                       validation_data = valid_batch, 
                       validation_steps=len(valid_batch), callbacks=[tensorboard_callback] ) #, hparams_callback])
# record time data
endtime = time.perf_counter()
pzastk4dur = round(endtime - starttime,2)
df4 = pd.DataFrame(pzaHist4.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
df4.rename(columns = {'index':'epochs'}, inplace=True)
df4
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
import json
type(pzastk4.get_config().items())
stf = pzastk4.get_config().items()
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

mdltxt = pzastk4.summary()
stringlist = []
pzastk4.summary(print_fn=lambda x: stringlist.append(x))
pzastk4mdlsum = "\n".join(stringlist)
pzastk4mdlsum = pzastk4mdlsum.replace('_________________________________________________________________\n', '')


#%% ChartMnistNetworkChanges
ChartMnistNetworkChanges(df3, df4, pzastk3mdlsum, pzastk3dur, pzastk4dur,
                         pzastk4mdlsum, 'Augmented Data', 
                         'Udemy TensorFlowCertZtoM COMPARE CNNs on PizzaSteak pics',
                         '3rd CNN Model',
                         '4th CNN Model PizzaSteak w//DataAugment'
                         )
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
#%%   # View our example image
import matplotlib.image as  mpimg
!c:\users\steve\wget.exe https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg 
steak = mpimg.imread("03-steak.jpeg")
plt.imshow(steak)
plt.axis(False);