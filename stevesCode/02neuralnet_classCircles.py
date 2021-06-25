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
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import make_circles

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
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(tf.version.GIT_VERSION, tf.version.VERSION)

from tensorflow.python import _pywrap_util_port
print("MKL enabled:", _pywrap_util_port.IsMklEnabled())


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
#%% visualize, visualize, visualize!
import numpy as np

def plot_decision_boundary(model = model1, X=X, y=y):
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
model_3.fit(X_reg_train, y_reg_train, epochs=150)
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
                    