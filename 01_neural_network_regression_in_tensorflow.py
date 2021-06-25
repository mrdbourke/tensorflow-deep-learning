#!/usr/bin/env python
# coding: utf-8
#%%
# <a href="https://colab.research.google.com/github/mrdbourke/tensorflow-deep-learning/blob/main/01_neural_network_regression_in_tensorflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Introduction to Regression with Neural Networks in TensorFlow Tutorial
# 
# There are many definitions for a [regression problem](https://en.wikipedia.org/wiki/Regression_analysis) but in our case, we're going to simplify it to be: predicting a number.
# 
# For example, you might want to:
# - Predict the selling price of houses given information about them (such as number of rooms, size, number of bathrooms).
# - Predict the coordinates of a bounding box of an item in an image.
# - Predict the cost of medical insurance for an individual given their demographics (age, sex, gender, race).
# 
# In this notebook, we're going to set the foundations for how you can take a sample of inputs (this is your data), build a neural network to discover patterns in those inputs and then make a prediction (in the form of a number) based on those inputs.
# 
# ## What we're going to cover
# 
# Specifically, we're going to go through doing the following with TensorFlow:
# - Architecture of a regression model
# - Input shapes and output shapes
#   - `X`: features/data (inputs)
#   - `y`: labels (outputs)
# - Creating custom data to view and fit
# - Steps in modelling
#   - Creating a model
#   - Compiling a model
#     - Defining a loss function
#     - Setting up an optimizer
#     - Creating evaluation metrics
#   - Fitting a model (getting it to find patterns in our data)
# - Evaluating a model
#   - Visualizng the model ("visualize, visualize, visualize")
#   - Looking at training curves
#   - Compare predictions to ground truth (using our evaluation metrics)
# - Saving a model (so we can use it later)
# - Loading a model
# 
# # Typical architecture of a regresison neural network
# 
# | **Hyperparameter** | **Typical value** |
# | --- | --- |
# | Input layer shape | Same shape as number of features (e.g. 3 for # bedrooms, # bathrooms, # car spaces in housing price prediction) |
# | Hidden layer(s) | Problem specific, minimum = 1, maximum = unlimited |
# | Neurons per hidden layer | Problem specific, generally 10 to 100 |
# | Output layer shape | Same shape as desired prediction shape (e.g. 1 for house price) |
# | Hidden activation | Usually [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) (rectified linear unit) |
# | Output activation | None, ReLU, logistic/tanh |
# | Loss function | [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) (mean square error) or [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) (mean absolute error)/Huber (combination of MAE/MSE) if outliers |
# | Optimizer | [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (stochastic gradient descent), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) |
# 
# > 🔑 **Note:** A **hyperparameter** in machine learning is something a data analyst or developer can set themselves, where as a **parameter** usually describes something a model learns on its own (a value not explicitly set by an analyst).
# 
# In[1]:


import tensorflow as tf
print(tf.__version__) # check the version (should be 2.x+)


# # Creating data to view and fit
# 
# Since we're working on a **regression problem** (predicting a number) let's create some linear data (a straight line) to model.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Create features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
plt.scatter(X, y);


# In[3]:  houseinfo...
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
house_info, house_price


# In[4]:
print(house_info.shape)

# In[5]:
import numpy as np
import matplotlib.pyplot as plt

# Create features (using tensors)
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels (using tensors)
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
plt.scatter(X, y);


# Our goal here will be to use `X` to predict `y`.
# 
# So our **input** will be `X` and our **output** will be `y`.
# 
# Knowing this, what do you think our input and output shapes will be?
# 
# Let's take a look.

# In[6]:


# Take a single example of X
input_shape = X[0].shape 

# Take a single example of y
output_shape = y[0].shape

print(input_shape, output_shape) # these are both scalars (no shape)

# Huh?
# From this it seems our inputs and outputs have no shape?
# How could that be?
# It's because no matter what kind of data we pass to our model, it's always going to take as input and return as ouput some kind of tensor.
# But in our case because of our dataset (only 2 small lists of numbers), we're looking at a special kind of tensor, more specificially a rank 0 tensor or a scalar.
# In[7]:
# Let's take a look at the single examples invidually
X[0], y[0]


# In our case, we're trying to build a model to predict the pattern between `X[0]` equalling `-7.0` and `y[0]` equalling `3.0`.
# 
# So now we get our answer, we're trying to use 1 `X` value to predict 1 `y` value.
# 
# You might be thinking, "this seems pretty complicated for just predicting a straight line...".
# 
# And you'd be right.
# 
# But the concepts we're covering here, the concepts of input and output shapes to a model are fundamental. 
# 
# In fact, they're probably two of the things you'll spend the most time on when you work with neural networks: **making sure your input and outputs are in the correct shape**.
# 
# If it doesn't make sense now, we'll see plenty more examples later on (soon you'll notice the input and output shapes can be almost anything you can imagine).
# 
# ![example of input and output shapes for a housing price prediction problem](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/01-input-and-output-shapes-housing-prices.png)
# *If you were working on building a machine learning algorithm for predicting housing prices, your inputs may be number of bedrooms, number of bathrooms and number of garages, giving you an input shape of 3 (3 different features). And since you're trying to predict the price of the house, your output shape would be 1.*

# # Steps in modelling
# 
# Now we know what data we have as well as the input and output shapes, let's see how we'd build a neural network to model it.
# 
# In TensorFlow, there are typically 3 fundamental steps to creating and training a model.
# 
# 1. **Creating a model** - piece together the layers of a neural network yourself (using the [Functional](https://www.tensorflow.org/guide/keras/functional) or [Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)) or import a previously built model (known as transfer learning).
# 2. **Compiling a model** - defining how a models performance should be measured (loss/metrics) as well as defining how it should improve (optimizer). 
# 3. **Fitting a model** - letting the model try to find patterns in the data (how does `X` get to `y`). 
# 
# Let's see these in action using the [Keras Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) to build a model for our regression data. And then we'll step through each.
# 

# In[8]:


# tf.Session(config=tf.ConfigProto(device_count = {'GPU': 1}))


# In[9]:


# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])

# Fit the model
model.fit(X, y, epochs=5)


# Boom!
# 
# We've just trained a model to figure out the patterns between `X` and `y`.
# 
# How do you think it went?

# In[10]:


# Check out X and y
X, y
# What do you think the outcome should be if we passed our model an `X` value of 17.0?
# In[11]:# Make a prediction with the model
model.predict([17.0])

# It doesn't go very well... it should've output something close to 27.0.
# > 🤔 **Question:** What's Keras? I thought we were working with TensorFlow but every time we write TensorFlow code, `keras` comes after `tf` (e.g. `tf.keras.layers.Dense()`)?
# Before TensorFlow 2.0+, [Keras](https://keras.io/) was an API designed to be able to build deep learning models with ease. Since TensorFlow 2.0+, its functionality has been tightly integrated within the TensorFlow library.
# # Improving a model
# How do you think you'd improve upon our current model?
# If you guessed by tweaking some of the things we did above, you'd be correct.
# To improve our model, we alter almost every part of the 3 steps we went through before.
# 1. **Creating a model** - here you might want to add more layers, increase the number of hidden units (also called neurons) within each layer, change the activation functions of each layer.
# 2. **Compiling a model** - you might want to choose optimization function or perhaps change the **learning rate** of the optimization function.
# 3. **Fitting a model** - perhaps you could fit a model for more **epochs** (leave it training for longer) or on more data (give the model more examples to learn from).
# 
# Woah. We just introduced a bunch of possible steps. The important thing to remember is how you alter each of these will depend on the problem you're working on.
# 
# And the good thing is, over the next few problems, we'll get hands-on with all of them.
# 
# For now, let's keep it simple, all we'll do is train our model for longer (everything else will stay the same).

# In[12]: # Set random seed & create model
tf.random.set_seed(42)
# Create a model (same as above)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile model (same as above)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
#%%  Fit model (this time we'll train for longer)
model.fit(X, y, epochs=100) # train for 100 epochs not 10
# You might've noticed the loss value decrease from before (and keep decreasing as the number of epochs gets higher). 
# What do you think this means for when we make a prediction with our model?
# How about we try predict on 17.0 again?

# In[13]:
# Remind ourselves of what X and y are
X, y


# In[14]: # Try and predict what y would be if X was 17.0
model.predict([17.0]) # the right answer is 27.0 (y = X + 10)
# Much better! 
# 
# We got closer this time. But we could still be better.
# Now we've trained a model, how could we evaluate it?
# # Evaluating a model 
# A typical workflow you'll go through when building neural networks is:
# ```
# Build a model -> evaluate it -> build (tweak) a model -> evaulate it -> build (tweak) a model -> evaluate it...
# ```
# The tweaking comes from maybe not building a model from scratch but adjusting an existing one.
# ### Visualize, visualize, visualize
# When it comes to evaluation, you'll want to remember the words: "visualize, visualize, visualize." 
# This is because you're probably better looking at something (doing) than you are thinking about something.
# It's a good idea to visualize:
# * **The data** - what data are you working with? What does it look like?
# * **The model itself** - what does the architecture look like? What are the different shapes?
# * **The training of a model** - how does a model perform while it learns?
# * **The predictions of a model** - how do the predictions of a model line up against the ground truth (the original labels)?
# 
# Let's start by visualizing the model.
# 
# But first, we'll create a little bit of a bigger dataset and a new model we can use (it'll be the same as before, but the more practice the better).
# 

# In[15]:


# Make a bigger dataset
X = np.arange(-100, 100, 4)
X
# In[16]:
# Make labels for the dataset (adhering to the same pattern as before)
y = np.arange(-90, 110, 4)
y


# Since $y=X+10$, we could make the labels like so:

# In[17]:


# Same result as above
y = X + 10
y


# ## Split data into training/test set
# 
# One of the other most common and important steps in a machine learning project is creating a training and test set (and when required, a validation set).
# 
# Each set serves a specific purpose:
# * **Training set** - the model learns from this data, which is typically 70-80% of the total data available (like the course materials you study during the semester).
# * **Validation set** - the model gets tuned on this data, which is typically 10-15% of the total data available (like the practice exam you take before the final exam).
# * **Test set** - the model gets evaluated on this data to test what it has learned, it's typically 10-15% of the total data available (like the final exam you take at the end of the semester).
# 
# For now, we'll just use a training and test set, this means we'll have a dataset for our model to learn on as well as be evaluated on.
# 
# We can create them by splitting our `X` and `y` arrays.
# 
# > 🔑 **Note:** When dealing with real-world data, this step is typically done right at the start of a project (the test set should always be kept separate from all other data). We want our model to learn on training data and then evaluate it on test data to get an indication of how well it **generalizes** to unseen examples.

# In[18]:


# Check how many samples we have
len(X)

# In[19]:
# Split data into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]

len(X_train), len(X_test)
# ## Visualizing the data
# Now we've got our training and test data, it's a good idea to visualize it.
# Let's plot it with some nice colours to differentiate what's what.
# In[20]:
plt.figure(figsize=(10, 7))
# Plot training data in blue
plt.scatter(X_train, y_train, c='b', label='Training data')
# Plot test data in green
plt.scatter(X_test, y_test, c='g', label='Testing data')
# Show the legend
plt.legend();


# Beautiful! Any time you can visualize your data, your model, your anything, it's a good idea. 
# 
# With this graph in mind, what we'll be trying to do is build a model which learns the pattern in the blue dots (`X_train`) to draw the green dots (`X_test`).
# 
# Time to build a model. We'll make the exact same one from before (the one we trained for longer).

# In[21]:
# Set random seed
tf.random.set_seed(42)
# Create a model (same as above)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile model (same as above)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
# Fit model (same as above)
#model.fit(X_train, y_train, epochs=100) # commented out on purpose (not fitting it just yet)
# ## Visualizing the model
# After you've built a model, you might want to take a look at it (especially if you haven't built many before).
# You can take a look at the layers and shapes of your model by calling [`summary()`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary) on it.
# > 🔑 **Note:** Visualizing a model is particularly helpful when you run into input and output shape mismatches.

# In[22]:
# Doesn't work (model not fit/built)
model.summary()
# Ahh, the cell above errors because we haven't fit our built our model.
# We also haven't told it what input shape it should be expecting.
# Remember above, how we discussed the input shape was just one number?
# We can let our model know the input shape of our data using the `input_shape` parameter to the first layer (usually if `input_shape` isn't defined, Keras tries to figure it out automatically).
# In[23]:
# Set random seed
tf.random.set_seed(42)

# Create a model (same as above)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1 , input_shape=[1]) # define the input_shape to our model
])

# Compile model (same as above)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
# In[24]:
# This will work after specifying the input shape
model.summary()
# Calling `summary()` on our model shows us the layers it contains, the output shape and the number of parameters.
# * **Total params** - total number of parameters in the model.
# * **Trainable parameters** - these are the parameters (patterns) the model can update as it trains.
# * **Non-trainable parameters** - these parameters aren't updated during training (this is typical when you bring in the already learned patterns from other models during transfer learning).
# 
# > 📖 **Resource:** For a more in-depth overview of the trainable parameters within a layer, check out [MIT's introduction to deep learning video](https://youtu.be/njKP3FqW3Sk).
# 
# > 🛠 **Exercise:** Try playing around with the number of hidden units in the `Dense` layer (e.g. `Dense(2)`, `Dense(3)`). How does this change the Total/Trainable params? Investigate what's causing the change.
# 
# For now, all you need to think about these parameters is that their learnable patterns in the data.
# 
# Let's fit our model to the training data.

# In[25]:  # Fit the model to the training data
model.fit(X_train, y_train, epochs=100, verbose=0) # verbose controls how much gets output
# In[26]:# Check the model summary
model.summary()
# Alongside summary, you can also view a 2D plot of the model using [`plot_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model).
# In[27]:
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, rankdir='TB', dpi=192, expand_nested=True)
# In our case, the model we used only has an input and an output but visualizing more complicated models can be very helpful for debugging.
# ## Visualizing the predictions
# Now we've got a trained model, let's visualize some predictions.
# To visualize predictions, it's always a good idea to plot them against the ground truth labels.
# Often you'll see this in the form of `y_test` vs. `y_pred` (ground truth vs. predictions).
# First, we'll make some predictions on the test data (`X_test`), remember the model has never seen the test data.
# In[28]:# Make predictions
y_preds = model.predict(X_test)
# In[29]:
# View the predictions
y_preds
# Okay, we get a list of numbers but how do these compare to the ground truth labels?
# 
# Let's build a plotting function to find out.
# 
# > 🔑 **Note:** If you think you're going to be visualizing something a lot, it's a good idea to functionize it so you can use it later.

# In[30]:
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=y_preds):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend();
# In[31]:
plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)


# From the plot we can see our predictions aren't totally outlandish but they definitely aren't anything special either.

# ## Evaluating predictions
# 
# Alongisde visualizations, evaulation metrics are your alternative best option for evaluating your model.
# 
# Depending on the problem you're working on, different models have different evaluation metrics. 
# 
# Two of the main metrics used for regression problems are:
# * **Mean absolute error (MAE)** - the mean difference between each of the predictions.
# * **Mean squared error (MSE)** - the squared mean difference between of the predictions (use if larger errors are more detrimental than smaller errors).
# 
# The lower each of these values, the better.
# 
# You can also use [`model.evaluate()`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate) which will return the loss of the model as well as any metrics setup during the compile step.
# In[32]:
# Evaluate the model on the test set
model.evaluate(X_test, y_test)
# In our case, since we used MAE for the loss function as well as MAE for the metrics, `model.evaulate()` returns them both.
# TensorFlow also has built in functions for MSE and MAE.
# For many evaluation functions, the premise is the same: compare predictions to the ground truth labels.
# In[33]:# Calculate the mean absolute error
mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=y_preds)
mae
# Huh? That's strange, MAE should be a single output.
# Instead, we get 10 values.
# This is because our `y_test` and `y_preds` tensors are different shapes.
# In[34]:# Check the test label tensor values
y_test
# In[35]:# Check the predictions tensor values (notice the extra square brackets)
y_preds
# In[36]:# Check the tensor shapes
y_test.shape, y_preds.shape
# Remember how we discussed dealing with different input and output shapes is one the most common issues you'll come across, this is one of those times.
# But not to worry.
# We can fix it using [`squeeze()`](https://www.tensorflow.org/api_docs/python/tf/squeeze), it'll remove the the `1` dimension from our `y_preds` tensor, making it the same shape as `y_test`.
# > 🔑 **Note:** If you're comparing two tensors, it's important to make sure they're the right shape(s) (you won't always have to manipulate the shapes, but always be on the look out, *many* errors are the result of mismatched tensors, especially mismatched input and output shapes).
# In[37]:# Shape before squeeze()
y_preds.shape
# In[38]:# Shape after squeeze()
y_preds.squeeze().shape
# In[39]:# What do they look like?
y_test, y_preds.squeeze()
# Okay, now we know how to make our `y_test` and `y_preds` tenors the same shape, let's use our evaluation metrics.
# In[40]:# Calcuate the MAE
mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=y_preds.squeeze()) # use squeeze() to make same shape
mae
# In[41]:# Calculate the MSE
mse = tf.metrics.mean_squared_error(y_true=y_test,
                                    y_pred=y_preds.squeeze())
mse
# We can also calculate the MAE using pure TensorFlow functions.
# In[42]:# Returns the same as tf.metrics.mean_absolute_error()
tf.reduce_mean(tf.abs(y_test-y_preds.squeeze()))
# Again, it's a good idea to functionize anything you think you might use over again (or find yourself using over and over again).
# 
# Let's make functions for our evaluation metrics.

# In[43]:
def mae(y_test, y_pred):
  """
  Calculuates mean absolute error between y_test and y_preds.
  """
  return tf.metrics.mean_absolute_error(y_test,
                                        y_pred)
  
def mse(y_test, y_pred):
  """
  Calculates mean squared error between y_test and y_preds.
  """
  return tf.metrics.mean_squared_error(y_test,
                                       y_pred)
# ## Running experiments to improve a model
# After seeing the evaluation metrics and the predictions your model makes, it's likely you'll want to improve it.
# Again, there are many different ways you can do this, but 3 of the main ones are:
# 1. **Get more data** - get more examples for your model to train on (more opportunities to learn patterns).
# 2. **Make your model larger (use a more complex model)** - this might come in the form of more layers or more hidden units in each layer.
# 3. **Train for longer** - give your model more of a chance to find the patterns in the data.
# Since we created our dataset, we could easily make more data but this isn't always the case when you're working with real-world datasets.
# So let's take a look at how we can improve our model using 2 and 3.
# To do so, we'll build 3 models and compare their results:
# 1. `model_1` - same as original model, 1 layer, trained for 100 epochs.
# 2. `model_2` - 2 layers, trained for 100 epochs.
# 3. `model_3` - 2 layers, trained for 500 epochs.
# 
# **Build `model_1`**

# In[44]: # Replicate original model
# Set random seed
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])
# Fit the model
model_1.fit(X_train, y_train, epochs=100, verbose=0)
# In[45]:# Make and plot predictions for model_1
y_preds_1 = model_1.predict(X_test)
plot_predictions(predictions=y_preds_1)
# In[46]:# Calculate model_1 metrics
mae_1 = mae(y_test, y_preds_1.squeeze()).numpy()
mse_1 = mse(y_test, y_preds_1.squeeze()).numpy()
print(mae_1, mse_1)
# In[47]: # multiple runs on this version ONLY... to find Adam(.04) as pretty good!
# **Build `model_2`**
# This time we'll add an extra dense layer (so now our model will have 2 layers) whilst keeping everything else the same.
# Set random seed
tf.random.set_seed(42)
model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(50, input_shape=[1], name='SteveInput'),
  tf.keras.layers.Dense(1) # add a second layer
], name ='TwohiddenwithAdam')

# Compile the model
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(.04),
                metrics=['mae'])

# Fit the model
model_2.fit(X_train, y_train, epochs=100, verbose=0) # set verbose to 0 for less output

y_preds_2 = model_2.predict(X_test)
plot_predictions(predictions=y_preds_2)
# Woah, that's looking better already! And all it took was an extra layer.
# In[49]:# Calculate model_2 metrics
model_2.summary()
mae_2 = mae(y_test, y_preds_2.squeeze()).numpy()
mse_2 = mse(y_test, y_preds_2.squeeze()).numpy()
print(mae_2, mse_2)
plot_model(model_2, show_shapes=True, rankdir='TB', dpi=192, expand_nested=True)

#%% In[47]: # Replicate model_1 and add an extra layer ** Big improvement **
# **Build `model_2a`**
# This time we'll switch to Adam optimizer whilst keeping everything else the same.
# Set random seed
tf.random.set_seed(42)
model_2a = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(1) # add a second layer
])

# Compile the model
model_2a.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model
model_2a.fit(X_train, y_train, epochs=100, verbose=0) # set verbose to 0 for less output
# In[48]:# Make and plot predictions for model_2
model_2a.summary()
y_preds_2a = model_2a.predict(X_test)
plot_predictions(predictions=y_preds_2a)
plot_model(model_2a, show_shapes=True, rankdir='TB', dpi=192, expand_nested=True)


# **Build `model_3`**
# For our 3rd model, we'll keep everything the same as `model_2` except this time we'll train for longer (500 epochs instead of 100).
# This will give our model more of a chance to learn the patterns in the data.
# In[50]: Fit the model (this time for 500 epochs, not 100)
# Set random seed
tf.random.set_seed(42)

# Replicate model_2
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1)
])

# Compile the model
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])
# Fit the model (this time for 500 epochs, not 100)
model_3.fit(X_train, y_train, epochs=500, verbose=0) # set verbose to 0 for less output
# In[51]: # Make and plot predictions for model_3
y_preds_3 = model_3.predict(X_test)
plot_predictions(predictions=y_preds_3)
# Strange, we trained for longer but our model performed worse?
# As it turns out, our model might've trained too long and has thus resulted in worse results (we'll see ways to prevent training for too long later on).
# In[52]:# Calculate model_3 metrics
mae_3 = mae(y_test, y_preds_3.squeeze()).numpy()
mse_3 = mse(y_test, y_preds_3.squeeze()).numpy()
mae_3, mse_3
# ## Comparing results
# 
# Now we've got results for 3 similar but slightly different results, let's compare them.

# In[53]:
model_results = [["model_1", mae_1, mse_1],
                 ["model_2", mae_2, mse_2],
                 ["model_3", mae_3, mse_3]]
# In[54]:
import pandas as pd
all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
all_results
# From our experiments, it looks like `model_2` performed the best.
# And now, you might be thinking, "wow, comparing models is tedious..." and it definitely can be, we've only compared 3 models here. 
# But this is part of what machine learning modelling is about, trying many different combinations of models and seeing which performs best.
# Each model you build is a small experiment. 
# > 🔑 **Note:** One of your main goals should be to minimize the time between your experiments. The more experiments you do, the more things you'll figure out which don't work and in turn, get closer to figuring out what does work. Remember the machine learning practitioner's motto: "experiment, experiment, experiment".
# Another thing you'll also find is what you thought may work (such as training a model for longer) may not always work and the exact opposite is also often the case.
# ## Tracking your experiments
# One really good habit to get into is tracking your modelling experiments to see which perform better than others.
# We've done a simple version of this above (keeping the results in different variables).
# > 📖 **Resource:** But as you build more models, you'll want to look into using tools such as:
# * [**TensorBoard**](https://tensorboard.dev/) - a component of the TensorFlow library to help track modelling experiments (we'll see this later).
# * [**Weights & Biases**](https://www.wandb.com/) - a tool for tracking all kinds of machine learning experiments (the good news for Weights & Biases is it plugs into TensorBoard).
# # Saving a model
# Once you've trained a model and found one which performs to your liking, you'll probably want to save it for use elsewhere (like a web application or mobile device).
# You can save a TensorFlow/Keras model using [`model.save()`](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model).
# There are two ways to save a model in TensorFlow:
# 1. The [SavedModel format](https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format) (default).
# 2. The [HDF5 format](https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format).
# The main difference between the two is the SavedModel is automatically able to save custom objects (such as special layers) without additional modifications when loading the model back in.
# Which one should you use?
# It depends on your situation but the SavedModel format will suffice most of the time.
# Both methods use the same method call.
# In[55]: # Save a model using the SavedModel format
model_2.save('best_model_SavedModel_format')
# In[58]: # Check it out - outputs a protobuf binary file (.pb) as well as other files
get_ipython().system('dir best_model_SavedModel_format')
# Now let's save the model in the HDF5 format, we'll use the same method but with a different filename.
# In[59]:  # Save a model using the HDF5 format
model_2.save("best_model_HDF5_format.h5") # note the addition of '.h5' on the end
# In[60]: # Check it out
get_ipython().system('dir best_model_HDF5_format.h5')

# # Loading a model
# 
# We can load a saved model using the [`load_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model) method.
# 
# Loading a model for the different formats (SavedModel and HDF5) is the same (as long as the pathnames to the particuluar formats are correct).
# In[61]:  Load a model from the SavedModel format
loaded_saved_model = tf.keras.models.load_model("best_model_SavedModel_format")
loaded_saved_model.summary()
# Now let's test it out.
# In[62]: # Compare model_2 with the SavedModel version (should return True)
model_2_preds = model_2.predict(X_test)
saved_model_preds = loaded_saved_model.predict(X_test)
mae(y_test, saved_model_preds.squeeze()).numpy() == mae(y_test, model_2_preds.squeeze()).numpy()
# Loading in from the HDF5 is much the same.
# In[63]:
# Load a model from the HDF5 format
loaded_h5_model = tf.keras.models.load_model("best_model_HDF5_format.h5")
loaded_h5_model.summary()
# In[64]:
# Compare model_2 with the loaded HDF5 version (should return True)
h5_model_preds = loaded_h5_model.predict(X_test)
mae(y_test, h5_model_preds.squeeze()).numpy() == mae(y_test, model_2_preds.squeeze()).numpy()
# # Downloading a model (from Google Colab)
# 
# Say you wanted to get your model from Google Colab to your local machine, you can do one of the following things:
# * Right click on the file in the files pane and click 'download'.
# * Use the code below.
# In[ ]:  # Download the model (or any file) from Google Colab
from google.colab import files
files.download("best_model_HDF5_format.h5")
# # A larger example
# 
# Alright, we've seen the fundamentals of building neural network regression models in TensorFlow.
# 
# Let's step it up a notch and build a model for a more feature rich datase.
# 
# More specifically we're going to try predict the cost of medical insurance for individuals based on a number of different parameters such as, `age`, `sex`, `bmi`, `children`, `smoking_status` and `residential_region`.
# 
# To do, we'll leverage the pubically available [Medical Cost dataset](https://www.kaggle.com/mirichoi0218/insurance) available from Kaggle and [hosted on GitHub](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv).
# 
# > 🔑 **Note:** When learning machine learning paradigms, you'll often go through a series of foundational techniques and then practice them by working with open-source datasets and examples. Just as we're doing now, learn foundations, put them to work with different problems. Every time you work on something new, it's a good idea to search for something like "problem X example with Python/TensorFlow" where you substitute X for your problem.

# In[65]:  # Import required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# In[66]:
# Read in the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
# In[67]:
# Check out the insurance dataset
insurance.head()
# We're going to have to turn the non-numerical columns into numbers (because a neural network can't handle non-numerical inputs).
# 
# To do so, we'll use the [`get_dummies()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) method in pandas.
# 
# It converts categorical variables (like the `sex`, `smoker` and `region` columns) into numerical variables using one-hot encoding.
# In[68]: # Turn all categories into numbers
insurance_one_hot = pd.get_dummies(insurance)
insurance_one_hot.head() # view the converted columns
# Now we'll split data into features (`X`) and labels (`y`).
# In[69]:
# Create X & y values
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]
# In[70]:# View features
X.head()
# And create training and test sets. We could do this manually, but to make it easier, we'll leverage the already available [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function available from Scikit-Learn.
# In[71]: # Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42) # set random state for reproducible splits

# Now we can build and fit a model (we'll make it the same as `model_2`).
# In[72]:
# Set random seed
tf.random.set_seed(42)
# Create a new model (same as model_2)
insurance_model = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1)
])
# Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=['mae'])

# Fit the model
insurance_model.fit(X_train, y_train, epochs=100)
# In[73]:  # Check the results of the insurance model
insurance_model.evaluate(X_test, y_test)
# Our model didn't perform very well, let's try a bigger model.
# We'll try 3 things:
# - Increasing the number of layers (2 -> 3).
# - Increasing the number of units in each layer (except for the output layer).
# - Changing the optimizer (from SGD to Adam).
# Everything else will stay the same.
# In[80]: # Set random seed
tf.random.set_seed(42)
# Add an extra layer and increase number of units
insurance_model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(100), # 100 units
  tf.keras.layers.Dense(10), # 10 units
  tf.keras.layers.Dense(1) # 1 unit (important for output layer)
])

# Compile the model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(), # Adam works but SGD doesn't 
                          metrics=['mae'])

# Fit the model and save the history (we can plot this)
history = insurance_model_2.fit(X_train, y_train, epochs=100, verbose=0)
# In[81]: # Evaluate our larger model
insurance_model_2.evaluate(X_test, y_test)
insurance_model_2.summary()
# Much better! Using a larger model and the Adam optimizer results in almost half the error as the previous model.
# > 🔑 **Note:** For many problems, the [Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) is a great starting choice. See Andrei Karpathy's "Adam is safe" point from [*A Recipe for Training Neural Networks*](http://karpathy.github.io/2019/04/25/recipe/) for more. 
# Let's check out the loss curves of our model, we should see a downward trend.
# In[82]: # Plot history (also known as a loss curve)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs");
# From this, it looks like our model's loss (and MAE) were both still decreasing (in our case, MAE and loss are the same, hence the lines in the plot overlap eachother).
# 
# What this tells us is the loss might go down if we try training it for longer.
# 
# > 🤔 **Question:** How long should you train for? 
# 
# > It depends on what problem you're working on. Sometimes training won't take very long, other times it'll take longer than you expect. A common method is to set your model training for a very long time (e.g. 1000's of epochs) but set it up with an [EarlyStopping callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) so it stops automatically when it stops improving. We'll see this in another module.
# 
# Let's train the same model as above for a little longer. We can do this but calling fit on it again.

# In[83]:  # Try training for a little longer (100 more epochs)
history_2 = insurance_model_2.fit(X_train, y_train, epochs=200, verbose=0)
# How did the extra training go?
# In[84]: # Evaluate the model trained for 200 total epochs
insurance_model_2_loss, insurance_model_2_mae = insurance_model_2.evaluate(X_test, y_test)
insurance_model_2_loss, insurance_model_2_mae
# Boom! Training for an extra 100 epochs we see about a 10% decrease in error.
# 
# How does the visual look?

# In[85]: # Plot the model trained for 200 total epochs loss curves
pd.DataFrame(history_2.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs"); # note: epochs will only show 100 since we overrid the history variable
# ## Preprocessing data (normalization and standardization)
# A common practice when working with neural networks is to make sure all of the data you pass to them is in the range 0 to 1.
# This practice is called **normalization** (scaling all values from their original range to, e.g. between 0 and 100,000 to be between 0 and 1).
# There is another process call **standardization** which converts all of your data to unit variance and 0 mean.
# 
# These two practices are often part of a preprocessing pipeline (a series of functions to prepare your data for use with neural networks).
# 
# Knowing this, some of the major steps you'll take to preprocess your data for a neural network include:
# * Turning all of your data to numbers (a neural network can't handle strings).
# * Making sure your data is in the right shape (verifying input and output shapes).
# * [**Feature scaling**](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler):
#     * Normalizing data (making sure all values are between 0 and 1). This is done by subtracting the minimum value then dividing by the maximum value minus the minmum. This is also referred to as min-max scaling.
#     * Standardization (making sure all values have a mean of 0 and a variance of 1). This is done by substracting the mean value from the target feature and then dividing it by the standard deviation.
#     * Which one should you use?
#       * **With neural networks you'll tend to favour normalization** as they tend to prefer values between 0 and 1 (you'll see this espcially with image processing), however, you'll often find a neural network can perform pretty well with minimal feature scaling.
# 
# > 📖 **Resource:** For more on preprocessing data, I'd recommend reading the following resources:
# * [Scikit-Learn's documentation on preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-data).
# * [Scale, Standardize or Normalize with Scikit-Learn by Jeff Hale](https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02).
# 
# We've already turned our data into numbers using `get_dummies()`, let's see how we'd normalize it as well.

# In[86]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Read in the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")


# In[87]:


# Check out the data
insurance.head()


# Now, just as before, we need to transform the non-numerical columns into numbers and this time we'll also be normalizing the numerical columns with different ranges (to make sure they're all between 0 and 1).
# 
# To do this, we're going to use a few classes from Scikit-Learn:
# * [`make_column_transformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html) - build a multi-step data preprocessing function for the folllowing trnasformations:
#   * [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) - make sure all numerical columns are normalized (between 0 and 1).
#   * [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) - one hot encode the non-numerical columns.
# 
# Let's see them in action.

# In[88]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Create column transformer (this will help us normalize/preprocess our data)
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Create X & y
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Build our train and test sets (use random state to ensure same split as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit column transformer on the training data only (doing so on test data would result in data leakage)
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScalar) and one hot encoding (OneHotEncoder)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)


# Now we've normalized it and one-hot encoding it, what does our data look like now?

# In[89]:


# Non-normalized and non-one-hot encoded data example
X_train.loc[0]


# In[90]:


# Normalized and one-hot encoded example
X_train_normal[0]


# How about the shapes?

# In[91]:


# Notice the normalized/one-hot encoded shape is larger because of the extra columns
X_train_normal.shape, X_train.shape


# Our data is normalized and numerical, let's model it.
# 
# We'll use the same model as `insurance_model_2`.

# In[92]:


# Set random seed
tf.random.set_seed(42)

# Build the model (3 layers, 100, 10, 1 units)
insurance_model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])

# Fit the model for 200 epochs (same as insurance_model_2)
insurance_model_3.fit(X_train_normal, y_train, epochs=200, verbose=0) 


# Let's evaluate the model on normalized test set.

# In[93]:


# Evaulate 3rd model
insurance_model_3_loss, insurance_model_3_mae = insurance_model_3.evaluate(X_test_normal, y_test)


# And finally, let's compare the results from `insurance_model_2` (trained on non-normalized data) and `insurance_model_3` (trained on normalized data).

# In[94]:


# Compare modelling results from non-normalized data and normalized data
insurance_model_2_mae, insurance_model_3_mae


# From this we can see normalizing the data results in 10% less error using the same model than not normalizing the data.
# 
# This is **one of the main benefits of normalization: faster convergence time** (a fancy way of saying, your model gets to better results faster).
# 
# `insurance_model_2` may have eventually achieved the same results as `insurance_model_3` if we left it training for longer. 
# 
# Also, the results may change if we were to alter the architectures of the models, e.g. more hidden units per layer or more layers.
# 
# But since our main goal as neural network practioners is to decrease the time between experiments, anything that helps us get better results sooner is a plus.

# # 🛠 Exercises
# 
# We've a covered a whole lot pretty quickly.
# 
# So now it's time to have a **play around** with a few things and start to build up your initution.
# 
# I emphasise the words play around because that's very important. Try a few things out, run the code and see what happens.
# 
# 1. Create your own regression dataset (or make the one we created in "Create data to view and fit" bigger) and build fit a model to it.
# 2. Try building a neural network with 4 Dense layers and fitting it to your own regression dataset, how does it perform?
# 3. Try and improve the results we got on the insurance dataset, some things you might want to try include:
#   * Building a larger model (how does one with 4 dense layers go?).
#   * Increasing the number of units in each layer.
#   * Lookup the documentation of [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) and find out what the first parameter is, what happens if you increase it by 10x?
#   * What happens if you train for longer (say 300 epochs instead of 200)? 
# 4. Import the [Boston pricing dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/boston_housing/load_data) from TensorFlow [`tf.keras.datasets`](https://www.tensorflow.org/api_docs/python/tf/keras/datasets) and model it.
# 

# # 📖 Extra curriculum
# 
# If you're looking for extra materials relating to this notebook, I'd check out the following:
# 
# * [MIT introduction deep learning lecture 1](https://youtu.be/njKP3FqW3Sk) - gives a great overview of what's happening behind all of the code we're running.
# * Reading: 1-hour of [Chapter 1 of Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) by Michael Nielson - a great in-depth and hands-on example of the intuition behind neural networks.
# 
# To practice your regression modelling with TensorFlow, I'd also encourage you to look through [Lion Bridge's collection of datasets](https://lionbridge.ai/datasets/) or [Kaggle's datasets](https://www.kaggle.com/data), find a regression dataset which sparks your interest and try to model.
