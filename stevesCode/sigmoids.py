# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 08:41:03 2021

@author: steve
"""
import tensorflow as tf
import matplotlib.pyplot as plt

A = tf.cast(tf.range(-10, 10), tf.float32)
A

#see the tensor:
plt.plot(A)

# let's start by replicating sigmoid
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

sigmoid(A)
plt.plot(sigmoid(A))

def relu(x):
    return tf.maximum(x, 0)

plt.plot(relu(A))

relu(A)

