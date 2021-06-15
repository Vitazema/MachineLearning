#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets

# Prepare numerical data
n_pts = 500
centers = [[-1, 1], [-1, -1], [1, -1]]
X, y = datasets.make_blobs(n_samples=n_pts, centers=centers, cluster_std=0.4)
print(X, y)

# Plot different cluster points
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])

# %%
# Encode cluster points by using OneHotEncoder
from keras.utils.np_utils import to_categorical
y_category = to_categorical(y, 3)
print(y_category)

# %%
# Create deep multicategorial model and log to Tensorboard
import sys
sys.path.append("../")
from ml_utils import create_tensorboard_callback

# Configure the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3, input_shape=(2,), activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(X, y_category, verbose=0, batch_size=32, epochs=300, callbacks=[create_tensorboard_callback()])
# %%

# %%
