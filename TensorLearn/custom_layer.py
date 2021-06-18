#%%
# Prepare environment
import sys
import random
import numpy as np
#import tensorflow.experimental.numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.utils.vis_utils import plot_model
from PIL import Image, ImageFont, ImageDraw

from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Layer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

# Import datasets
from tensorflow.keras.datasets import fashion_mnist

# Load utils and configure GPU
sys.path.append("../")
from ml_utils import *
restrict_gpu_mem(7000)


#%%
class SimpDense(Layer):
  def __init__(self, units=32, activation=None):
    super(SimpDense, self).__init__()
    self.units = units
    self.activation = tf.keras.activations.get(activation)
  
  def build(self, input_shape): # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(name='kernel',
                        initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
                        trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(name='bias',
                        initial_value=b_init(shape=(self.units,), dtype='float32'),
                        trainable=True)

  def call(self, inputs): # Defines the computation from inputs to outputs
    return self.activation(tf.matmul(inputs, self.w) + self.b)


#%%
simp_dense = SimpDense(units=1)
x = tf.ones((1,1))
y = simp_dense(x)
print(simp_dense.variables)

#%% Hello World model
xs = np.array([-1., 0., 1., 2., 3., 4.], dtype=float)
ys = np.array([-3., -1., 1., 3., 5., 7.], dtype=float)

model = Sequential([simp_dense])
model.compile(optimizer='sgd', loss=tf.keras.losses.mean_squared_error)
model.fit(xs, ys, epochs=500, verbose=0)
print(model.predict([10.]))

# %%
# Fashion MNIST model
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
  Flatten(input_shape=(28,28)),
  SimpDense(128, activation = tf.nn.relu),
  Dropout(0.2),
  Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=Adam(0.001), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)
# %%
