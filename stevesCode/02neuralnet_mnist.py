# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:39:55 2021

@author: steve
"""
#%%
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
print(tf.__version__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2, random_state=42)
print(tf.version.GIT_VERSION, tf.version.VERSION)
from tensorflow.python import _pywrap_util_port
print("MKL enabled:", _pywrap_util_port.IsMklEnabled())
#%% Lecture 90 working with a larger example... multiclass classification.  fasion MNIST
'''Fashion-MNIST is a dataset of Zalando's article images consisting of a 
training set of 60,000 examples and a test set of 10,000 examples. 
Each example is a 28x28 grayscale image, associated with a label from 10 classes.
'''
from tensorflow.keras.datasets import fashion_mnist
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
train_data.shape
# Show the first training example
print(f"Training sample:\n{train_data[0]}\n") 
print(f"Training label: {train_labels[0]}")
# Check the shape of our data
train_data.shape, train_labels.shape, test_data.shape, test_labels.shape
train_data[0]
train_data[0].shape  # 28,28
type(train_data[0])  # ndarray
# Plot a single example
plt.imshow(train_data[12]);  # this shows an image in 'fake' color...the images are actually grayscale

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# How many classes are there (this'll be our output shape)?
len(class_names)

# Plot an example image and its label
plt.imshow(train_data[17], cmap=plt.cm.binary) # change the colours to black & white
plt.title(class_names[train_labels[17]]);

#%% 1st mnist model
starttime = time.perf_counter()
tf.random.set_seed(42)
mnist1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)) # this 'flattens the "square" picture to a "Flat vector"
    ,tf.keras.layers.Dense(28,'relu')
    ,tf.keras.layers.Dense(28,'relu')
    ,tf.keras.layers.Dense(10,'softmax')
    ])
mnist1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               optimizer = tf.keras.optimizers.Adam(),
               metrics=['accuracy'])
non_norm_history = mnist1.fit(train_data, train_labels, epochs=10,
                        validation_data=(test_data, test_labels)      )
endtime = time.perf_counter()
duration = round(endtime - starttime,2)

mnist1.summary() 
mnist1.evaluate(test_data, test_labels)
#%% normalize (get it between 0 & 1)
train_data_n = train_data/255.0
test_data_n = test_data/255.0
test_data.shape

starttime = time.perf_counter()

tf.random.set_seed(42)
mnist2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape= (28,28)),
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(10, 'softmax')
    ])
mnist2.compile(optimizer=tf.keras.optimizers.Adam(),
               loss=tf.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy']
               )
historyM2 = mnist2.fit(train_data_n, train_labels, epochs=10,
                       validation_data=(test_data_n, test_labels))
endtime = time.perf_counter()
duration_n = round(endtime - starttime,2)

mnist2.evaluate(test_data_n, test_labels)

df = pd.DataFrame(historyM2.history).reset_index()
df.rename(columns = {'index':'epochs'}, inplace=True)
df1 = pd.DataFrame(non_norm_history.history).reset_index()
df1.rename(columns = {'index':'epochs'}, inplace=True)
df
#%%  ChartMnistNetworkChanges
def ChartMnistNetworkChanges(dfLeft, dfRight, mdlsummary, dfLtime, dfRtime):
    modelstart = time.strftime('%c')
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.suptitle(f'Udemy TensorFlowCertZtoM Lect# 95 Optimizing NN on MNIST data {modelstart}')
    plt.title (f'Mnist Normalized {dfLtime:.2f} sec ')
    sns.lineplot(x='epochs', y='value', hue='variable', data = pd.melt(dfLeft,['epochs']))
    plt.text(0., .5,# transform=trans1,
              s=mdlsummary,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='pink', alpha=0.5))
    plt.legend(loc= 'lower left')
    plt.ylim(.3,.95)
    plt.xlabel('epochs    \GitHub\DeepLearning')

    plt.subplot(1,2,2)
    plt.text(-.3, .475,# transform=trans1,
              s=changes,
              wrap=True, ha='left', va='bottom', fontname='Consolas',
              fontsize=7, bbox=dict(facecolor='aqua', alpha=0.5))
    plt.title (f'Mnist Not normalized {dfRtime:.2f} sec ')
    sns.lineplot(x='epochs', y='value', hue='variable', data = pd.melt(dfRight,['epochs']))
    plt.ylim(.3,.95)
    plt.xlabel('02neuralnet_mnist.py    epochs')
    plt.show();
#%%
ChartMnistNetworkChanges(df, df1, short_model_summary, duration_n, duration)
short_model_summary
df1
#%% Document the model...
mdltxt = mnist3.summary()
stringlist = []
mnist3.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
print(short_model_summary)

#%% Now, find the best learning rate
starttime = time.perf_counter()

tf.random.set_seed(42)
mnist3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape= (28,28)),
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(10, 'softmax')
    ],name = 'mnist3')
mnist3.compile(optimizer=tf.keras.optimizers.Adam(),
               loss=tf.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy']
               )
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 **(epoch/10))
historyM3 = mnist3.fit(train_data_n, train_labels, epochs=10,
                       validation_data=(test_data_n, test_labels), 
                       callbacks=[lr_scheduler], verbose=1)
endtime = time.perf_counter()
duration_n = round(endtime - starttime,2)
df = pd.DataFrame(historyM3.history).reset_index()
df.drop('lr', axis=1, inplace=True)
df.rename(columns = {'index':'epochs'}, inplace=True)
df
mnist3.evaluate(test_data_n, test_labels)

#%% # Plot the learning rate versus the loss
lrs = 1e-3 * (10 ** (np.arange(20)/10))
lrs
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, historyM3.history["loss"]) # we want the x-axis (learning rate) to be log scale
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");

#%% 
changes = '''
Refit the model with the ideal learning rate... about .002
# step 2 drop the two middle layers to 4 nodes  ... acc: .78 time: 50.7 sec
# step 3 jump the nodes up to 32...             ... acc: .88 time: 51.2 sec
# step 4 jump the nodes up to 64...             ... acc: .89 time: 51.9 sec 
# step 5 jump the nodes up to 128...            ... acc: .87 time: 52.3 sec 
# step 6 add a network layer, return to 64      ... acc: .88 time: 62.6 sec
# step 7 drop down to a single relu layer       ... acc: .87 time: 44.8 sec
'''
starttime = time.perf_counter()

tf.random.set_seed(42)
mnist4 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape= (28,28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, 'softmax')
    ],name = 'mnist4_128nodes')
mnist4.compile(optimizer=tf.keras.optimizers.Adam(lr=.002),
               loss=tf.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy']
               )
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 **(epoch/10))
historyM4 = mnist4.fit(train_data_n, train_labels, epochs=10,
                       validation_data=(test_data_n, test_labels), 
                       #callbacks=[lr_scheduler], 
                       verbose=1)
endtime = time.perf_counter()
duration_n = round(endtime - starttime,2)
df = pd.DataFrame(historyM4.history).reset_index()
#df.drop('lr', axis=1, inplace=True)
df.rename(columns = {'index':'epochs'}, inplace=True)
df
#%% Document the model...
mdltxt = mnist4.summary()
stringlist = []
mnist4.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
print(short_model_summary)

mnist4.evaluate(test_data_n, test_labels)
ChartMnistNetworkChanges(df, df1, short_model_summary, duration_n, duration)
#%%  confustion matrix...
from sklearn.metrics import confusion_matrix
y_pred = mnist4.predict(test_data_n)
y_pred[0], tf.argmax(y_pred[0]), class_names[tf.argmax(y_pred[0])]
y_preds = tf.argmax(y_pred,axis=1) 
confusion_matrix(test_labels, y_preds)

con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=y_preds).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
classes=[0,1,2,3,4,5,6,7,8,9]
con_mat_df = pd.DataFrame(confusion_matrix(test_labels, tf.round(y_preds)),
                     index = classes, 
                     columns = classes)

figure = plt.figure(figsize=(8,8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues,annot_kws={"size":8},
            xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#%% 
y_pred[0], tf.argmax(y_pred[0]), class_names[tf.argmax(y_pred[0])]
y_preds = tf.argmax(y_pred,axis=1)

