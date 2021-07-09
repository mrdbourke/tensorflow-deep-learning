# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 08:39:29 2021

@author: steve
"""
# for below, had to install: tensorflow-mkl
# import mkl  I can't get this to load. don't know why.  23 Jun 2021
'''
https://github.com/tensorflow/tensorflow/issues/45853
The TF with MKL needs to be set optimization setting.
I try following setting in linux, got shorter time than stack TF in small_model (CNN).
But a little longer time than stack TF in medium model (LSTM).
Different model need different setting to reach best performance on CPU.
You could try to adjust it based on your CPU.

set TF_ENABLE_MKL_NATIVE_FORMAT=1  
set TF_NUM_INTEROP_THREADS=1
set TF_NUM_INTRAOP_THREADS=4
set  OMP_NUM_THREADS=4
set KMP_BLOCKTIME=1
set KMP_AFFINITY=granularity=fine,compact,1,0
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
print(tf.__version__)
from sklearn.datasets import make_circles
from datetime import datetime
from sklearn.model_selection import train_test_split
import time
n_samples = 1000
X,y = make_circles(n_samples, noise = .03, random_state=42)

X
y[:10]
X[:10]
import pandas as pd
circles = pd.DataFrame({"X0":X[:,0], 'X1':X[:,1], 'label':y})
circles.head()
circles.label.value_counts()

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)

X.shape, y.shape
len(X), len(y)
circles.head()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2, random_state=42)

print(tf.version.GIT_VERSION, tf.version.VERSION)

from tensorflow.python import _pywrap_util_port
print("MKL enabled:", _pywrap_util_port.IsMklEnabled())

# tf.get_logger().setLevel('INFO')
# tf.get_logger().setLevel('ERROR')  # THIS SEEMED TO WORK! :) 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# import logging
# tf.get_logger().setLevel(logging.ERROR)
#%% visualize, visualize, visualize!
import numpy as np

def plot_decision_boundary(model , X=X, y=y):
  """
  Plots the decision boundary created by a model predicting on X.
  This function has been adapted from two phenomenal resources:
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/madewithml/basics/blob/master/notebooks/09_Multilayer_Perceptrons/09_TF_Multilayer_Perceptrons.ipynb
  """
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  
  # Create X values (we're going to predict on all of these)
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html
  
  # Make predictions using the trained model
  y_pred = model.predict(x_in)

  # Check for multi-class
  if len(y_pred[0]) > 1:
    print("doing multiclass classification...")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(y_pred).reshape(xx.shape)
  
  # Plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
#  return y_pred[0]
 

#%% modeling
tf.random.set_seed(42)
#1 Create the model
model1 = tf.keras.Sequential([tf.keras.layers.Dense(100)
                             ,tf.keras.layers.Dense(1)
 ]                         )
#2 Compile the model
model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),  #For a CLASSIFICATION problem
               optimizer=tf.keras.optimizers.Adam(),
               metrics='accuracy')
#3 Fit the model (train)
model1.fit(X, y, epochs=100, verbose=0)
model1.evaluate(X,y)
plot_decision_boundary(model1, X, y)
#%% plot_decision_boundary
plot_decision_boundary()
#%%
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))

#%%  regression problem...
# Set random seed
tf.random.set_seed(42)

# Create some regression data
X_regression = np.arange(0, 1000, 5)
y_regression = np.arange(100, 1100, 5)

# Split it into training and test sets
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]
y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]
#%%
tf.random.set_seed(42)
model_3 = tf.keras.Sequential([tf.keras.layers.Dense(100),
                               tf.keras.layers.Dense(10),
                               tf.keras.layers.Dense(1)]
    )
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics = ['mae'])
model_3.fit(X_reg_train, y_reg_train, epochs=15)
# Make predictions with our trained model
y_reg_preds = model_3.predict(y_reg_test)

# Plot the model's predictions against our regression data
plt.figure(figsize=(10, 7))
plt.scatter(X_reg_train, y_reg_train, c='b', label='Training data')
plt.scatter(X_reg_test, y_reg_test, c='g', label='Testing data')
plt.scatter(X_reg_test, y_reg_preds.squeeze(), c='r', label='Predictions')
plt.legend();

#%%
tf.random.set_seed(42)
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)])
model_4.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        metrics=['accuracy'])
history = model_4.fit(X, y, epochs=15)

plot_decision_boundary(model_4, X, y)
#%% Implement the model from playground.tensorflow.org
tf.random.set_seed(42)
model5 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
# model5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
model5.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(lr=0.01),
               metrics = ['accuracy'])                   
history = model5.fit(X, y, epochs=19)
model5.evaluate(X, y)
plot_decision_boundary(model5, X, y)
#%%  Restart 1
n_samples = 1000
X,y = make_circles(n_samples, noise = .03, random_state=42)

len(X)
X, y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2, random_state=42)

X_train.shape, y_train.shape
#let's recreate model on training...test on test!! 
tf.random.set_seed(42)
model6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model6.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               metrics= ['accuracy'])
history = model6.fit(X_train, y_train, epochs=24)
model6.evaluate(X_test, y_test)
#%% Document the model...
mdltxt = model6.summary()
stringlist = []
model6.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
#%% visualize, visualize, visualize!
hist = pd.DataFrame(history.history)
modelstart = time.strftime('%c')
circleinput = '''In [329] circles.head()
Out[329]: 
         X0        X1  label
0  0.754246  0.231481      1
1 -0.756159  0.153259      1
2 -0.815392  0.173282      1
3 -0.393731  0.692883      1
4  0.442208 -0.896723      0'''

code = '''model6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model6.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               metrics= ['accuracy'])
history = model6.fit(X_train, y_train, epochs=24)
'''

neurons4reason = '''playground.tensorflow.org 'worked' with 2 layers, each
with 4 neurons.  That's where this model came from, and how the 
#of layers and neurons was selected.  serendipity.'''

plt.figure(figsize=(12,12))
plt.suptitle(f'Udemy TensorFlowCertZtoM Lect# 73 Using a NN to distinguish 2 cirles {modelstart}' +
             '\ncodeFile: \\github\\Deeplearning\\02neuralnet_classCircles.py')
plt.subplot(2,2,1)
plt.title('"created artificial" circle data and model')
plt.text(.1, .6, circleinput)
plt.text(.1, .1,# transform=trans1,
          s=short_model_summary,
          wrap=True, ha='left', va='bottom', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='pink', alpha=0.5))

plt.subplot(2,2,2)
plt.plot(hist.loss, label='Loss')
plt.plot(hist.accuracy, label='Accuracy')
plt.text(1,.2, s=code, ha='left', va='bottom', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='blue', alpha=0.2))
plt.text(1,.7, s=neurons4reason, ha='left', va='bottom', fontname='Consolas',
          fontsize=7, bbox=dict(facecolor='green', alpha=0.2))
plt.title('Model6 loss curves')
plt.legend()
plt.subplot(2,2,3)
plt.title('Training')
plot_decision_boundary(model6, X_train, y_train)
plt.subplot(2,2,4)
plt.title ('test')
plot_decision_boundary(model6, X_test, y_test)
plt.show();

#%% visualize the training history; plot the loss (training curves)
history.history
pd.DataFrame(history.history)
pd.DataFrame(history.history).plot()
plt.title('Model6 loss curves')
plt.show();

#%% lesson 84:  find the ideal learning rate...
# a learning rate callback.  TENSORBOARD!!!
#%%  # Define the Keras TensorBoard callback.
logdir="d:/data/logs/TFcertUdemy/" + datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
#%%  # Creates a file writer for the log directory.
imagedir = logdir+'/imgs'
file_writer = tf.summary.create_file_writer(imagedir)
#%% 
tf.random.set_seed(42)
model7 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu')
    ,tf.keras.layers.Dense(4, activation='relu')
    ,tf.keras.layers.Dense(1, activation='sigmoid')])
model7.compile(loss=tf.keras.losses.BinaryCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(learning_rate=.01),
               metrics=['accuracy'])
# create the LR callback...
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 **(epoch/20))

history7 = model7.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_callback,lr_scheduler], verbose=0)
model7.evaluate(X_train, y_train)
#%%

pd.DataFrame(history7.history).plot(figsize=(10,7), xlabel='epochs')
# Plot the learning rate versus the loss
lrs = 1e-4 * (10 ** (np.arange(100)/20))
lrs
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history7.history["loss"]) # we want the x-axis (learning rate) to be log scale
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");
#%%  Restart 2
# n_samples = 1000
# X,y = make_circles(n_samples, noise = .03, random_state=42)
# create the LR callback...
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 **(epoch/20))

logdir="d:/data/logs/TFcertUdemy/mdl8_" + datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
              histogram_freq=1,                                     
              profile_batch='500,520')  #this seemed to fix the errors noted in the opening dialog above. :) 
tf.random.set_seed(42)
model8 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model8.compile(loss='binary_crossentropy', #  tf.keras.losses.BinaryCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(lr=.02),
               metrics=['accuracy'])
history8 = model8.fit(X_train, y_train, epochs=20 , callbacks=[tensorboard_callback,lr_scheduler], verbose=0)
loss, accuracy = model8.evaluate(X_test, y_test)
print(f'Loss & Accuracy: {loss:.2f}, {accuracy*100:.2f}%')
print(f'Point tensorboard here: c:\\users\\steve>tensorboard --logdir {logdir}')
# Plot the decision boundaries for the training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model8, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model8, X=X_test, y=y_test)
plt.show()
#%%  confustion matrix...
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred = model8.predict(X_test) 
confusion_matrix(y_test, tf.round(y_pred))

con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
classes=[0,1]
con_mat_df = pd.DataFrame(confusion_matrix(y_test, tf.round(y_pred)),
                     index = classes, 
                     columns = classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues,annot_kws={"size":28})
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

