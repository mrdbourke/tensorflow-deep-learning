# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:09:59 2021

@author: steve
"""
#%% tps imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tensorflow as tf, sys
import numpy as np
#import tensorflow.experimental.numpy as np
import pandas as pd, tensorboard as tb, time, random
import matplotlib.image as mpimg, matplotlib.pyplot as plt
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
#from tensorflow.python.client import device_lib 
import seaborn as sns
from datetime import datetime
import zipfile
import wget

# %% Get GPU status
# print(device_lib.list_local_devices())  # this puts out a lot of lines (Gibberish?)
print('Conda Envronment:  ', os.environ['CONDA_DEFAULT_ENV'])
print(f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}')
print(f'Cuda Support:       {tf.test.is_built_with_cuda()}')
print(f'Tensor Flow:        {tf.version.VERSION}')
pver = str(format(sys.version_info.major) +'.'+ format(sys.version_info.minor)+'.'+ format(sys.version_info.micro))
print('Python version:      {}.'.format(pver)) 
print('The numpy version:   {}.'.format(np.__version__))
print('The panda version:   {}.'.format(pd.__version__))
print('Tensorboard version  {}.'.format(tb.__version__))
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
    plt.xlabel('05transferLearning02FineTuning.py    epochs')
    plt.legend(loc= 'lower left')
    plt.show();
# %% func: view_random_image 

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

# Create function to unzip a zipfile into current working directory 
# (since we're going to be downloading and unzipping a few files)
# %% def unzip_data
def unzip_data(zipfilepath, url ):
  """
  wgets url & Unzips the 'wgetted' file into the zipfilepath folder.

  Args:
    zipfilepath (str): a filepath to a target zip folder to be unzipped.
    url to the available zip file
    # sample call...
    # url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip'
    # targetfolder = 'd:/data/udemy/dbourkeTFcert'
    # unzip_data(targetfolder, url)
  """
  dwnldFile = wget.download(url, out=zipfilepath)  # this worked! :)
  zip_ref = zipfile.ZipFile(dwnldFile, "r")
  zip_ref.extractall(zipfilepath)
  zip_ref.close()
# %% ShowImageFolder_SetTrainTestDirs(imagelocationpath)
def ShowImageFolder_SetTrainTestDirs(imagelocationpath):
    '''
    Parameters
    ----------
    imagelocationpath : str
        sample: 'd:\\data\\udemy\\dbourkeTFcert\\10_Food_classes_10_percent'.

    Returns
    -------
    None.

    '''    
    for dirpath, dirnames, filenames in os.walk(imagelocationpath):
        print(f'There are {len(dirnames)} folders and {len(filenames)} images in {dirpath}')
    
    # setup the train and test directories...
    train_dir = imagelocationpath + '/train/'
    test_dir  = imagelocationpath + '/test/'
    
    import pathlib
    data_dir = pathlib.Path(train_dir) # turn our training path into a Python path
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
    print('class names are: ',class_names)
    
    view_random_image(train_dir, random.choice(class_names))
    return train_dir, test_dir

# %%  # Define the Keras TensorBoard callback.
# this must exist... logdir="d:/data/logs/TFcertUdemy/04food10cls/" 
def create_tb_callback(topdirname, dirname, expname):
    '''
    Parameters
    ----------
    topdirname: this has to exist
    dirname : str
        'tf_hub_src' middle folder (created by this function).
    expname : TYPE
        'resnet50v2' bottom folder (created by this function)
    Sample: tbcb = 
     create_tb_callback("d:/data/logs/TFcertUdemy/04food10cls/",'tf_hub_src', 'resnet50v2')

    Returns
    -------
    tensorboard_callback : TYPE
        tbcb.

    '''
#    log_dir = topdirname  + dirname + '\\' + expname+'_'  + datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = topdirname  + dirname + '\\' + expname+'_'  + datetime.now().strftime('%d_%H%M%S')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
    hparams_callback = hp.KerasCallback(topdirname, {'num_relu_units': 512,
                                    'dropout': 0.2})
    return tensorboard_callback, log_dir