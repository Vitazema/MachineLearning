#%%
import numpy as np
import matplotlib.pyplot as plt

# Define random points with std = 2
n_pts = 500
X_aliance = np.array([np.random.normal(13, 2, n_pts),
                np.random.normal(15, 2, n_pts)]).T
X_horde = np.array([np.random.normal(8, 2, n_pts),
              np.random.normal(7, 2, n_pts)]).T
# Summurize points into one array
X = np.vstack((X_aliance, X_horde))

# Label each point 0 and 1 (blue and red region)
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

# Plot graph
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

# %%
# Build and train model

import tensorflow as tf
import tensorboard
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import os 
current_dir_path = os.path.dirname(os.path.realpath(__file__))

# Configure tensorboard
logdir = os.path.join(current_dir_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
# categorical_crossentropy for multicategorial problem
model.compile(Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=400, callbacks=[tensorboard_callback])
# %%
def plot_decision_boundary(X, y, model):
  x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
  y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1)
  xx, yy = np.meshgrid(x_span, y_span)
  xx_, yy_ = xx.ravel(), yy.ravel()
  grid = np.c_[xx_, yy_]
  pred_func = model.predict(grid)
  z = pred_func.reshape(xx.shape)
  plt.contourf(xx, yy, z)

plot_decision_boundary(X, y, model)
# Plot graph
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
x = 7.5
y = 11.2
point = np.array([[x, y]])
manual_prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color='green')
print('prediciton is: ', manual_prediction)
# %%
