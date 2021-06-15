#%%
import numpy as np
#import tensorflow.experimental.numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#%%
import random

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Prepare environment
import sys

from tensorflow.python.keras.engine.training_utils import handle_partial_sample_weights
sys.path.append("../")
from ml_utils import *
restrict_gpu_mem(7000)

xs = np.array([-1., 0., 1., 2., 3., 4.], dtype=float)
ys = np.array([-3., -1., 1., 3., 5., 7.], dtype=float)

#%%
model = Sequential([
  Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss=tf.keras.losses.mean_squared_error)
model.fit(xs, ys, epochs=500, verbose=0)
print(model.predict([10.]))

# %%
# Define custom loss
def huber_loss(y_true, y_pred):
  threshold = 1
  error = y_true - y_pred
  is_small_error = tf.abs(error) <= threshold
  small_error_loss = tf.square(error) / 2
  big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
  return tf.where(is_small_error, small_error_loss, big_error_loss)

model = Sequential([
  Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss=huber_loss)
model.fit(xs, ys, epochs=500, verbose=0)
print(model.predict([10.]))

# %%
