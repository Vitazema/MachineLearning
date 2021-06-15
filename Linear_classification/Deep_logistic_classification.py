#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import datasets

n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, noise=0.1, factor=0.4)
print(X, y)

# %%
# Plot circular data
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])

#%%

from datetime import datetime
import os 
current_dir_path = os.path.dirname(os.path.realpath(__file__))

# Configure tensorboard
logdir = os.path.join(current_dir_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(2,), activation='sigmoid'))
# Need more than one hidden layer to correct (circular) shape decision boundary
model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation='sigmoid'))
model.add(tf.keras.layers.Dense(8, input_shape=(8,), activation='sigmoid'))
model.add(tf.keras.layers.Dense(4, input_shape=(8,), activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y,batch_size=32, verbose=0, epochs=300, callbacks=[tensorboard_callback])
# %%

def plot_decision_boundary(X, y, model):
  x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25, 50)
  y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25, 50)
  xx, yy = np.meshgrid(x_span, y_span)
  xx_, yy_ = xx.ravel(), yy.ravel()
  grid = np.c_[xx_, yy_]
  pred_func = model.predict(grid)
  z = pred_func.reshape(xx.shape)
  plt.contourf(xx, yy, z)

plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

#%%
# Check on manual data point prediciton
plot_decision_boundary(X, y, model)

plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

pX = 0.1
pY = 0.72
point = np.array([[pX, pY]])
prediction = model.predict(point)
plt.plot([pX], [pY], marker='o', markersize=10, color='red')
print('Prediction is: ', prediction)
# %%
