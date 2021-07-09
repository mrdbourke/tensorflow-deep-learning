# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 09:10:17 2021

@author: steve
"""

#%% Imports...
import time, sys, tensorflow as tf, tensorboard, sklearn.metrics, itertools, io
from datetime import datetime
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import pydot
import graphviz  #for graph of 
import GPUtil
gpus = GPUtil.getGPUs()

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
#%% # Get the class names (programmatically, this is much more helpful with a longer list of classes)
import pathlib
import numpy as np
data_dir = pathlib.Path("d:/data/udemy/dbourkeTFcert/pizza_steak/train/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)

#%% Walk through pizza_steak directory and list number of files
for dirpath, dirnames, filenames in os.walk('d:/data/udemy/dbourkeTFcert/pizza_steak/'):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
#%%  Read aan image...
import matplotlib.pyplot as plt
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
#%% # View a random image from the training dataset
img = view_random_image(target_dir="d:/data/udemy/dbourkeTFcert/pizza_steak/train/",
                        target_class="steak")
# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir('d:/data/udemy/dbourkeTFcert/pizza_steak/train/steak'))
num_steak_images_train

tf.constant(img)

#%%
# import zipfile
# import wget
# # https://www.pair.com/support/kb/paircloud-downloading-files-with-wget/
# # Download zip file of pizza_steak images
# # !wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip 
# url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip'
# output ='d:\\data\\udemy\\dbourkeTFcert'
# dwnldFile = wget.download(url, out=output)  # this worked! :)
# destination = os.path.join(output, dwnldFile)  # this join gives two dif slashes :(

# # Unzip the downloaded file
# zip_ref = zipfile.ZipFile("pizza_steak.zip", "r")
# zip_ref = zipfile.ZipFile(destination, "r")
# zipfile.ZipFile.namelist('d:\\data\\udemy\\dbourkeTFcert\\pizza_steak.zip')
# zipfile.ZipInfo.filename
# zip_ref.extractall()
# zip_ref.close()
#%%