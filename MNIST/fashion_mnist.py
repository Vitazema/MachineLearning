#%%
import numpy as np
#import tensorflow.experimental.numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#%%
import random
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Prepare environment
import sys

from tensorflow.python.keras.engine.training_utils import handle_partial_sample_weights
from tensorflow.python.keras.layers.core import Flatten
sys.path.append("../")
from ml_utils import *
restrict_gpu_mem(7000)

#%% Load in the MNIST data
fmnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fmnist.load_data()
X_train = X_train / 255.
X_test = X_test / 255.

model = Sequential([Flatten(input_shape=(28, 28)),
                    Dense(128, activation=tf.nn.relu),
                    Dense(10, activation=tf.nn.softmax)])

# Configure, train and evaluate the model
model.compile(optimizer=Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[create_tensorboard_callback('val')])           
# %%
