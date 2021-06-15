#%%
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras


points = 500
X = np.linspace(-3, 3, points)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)

plt.scatter(X, y)
# %%
model = keras.models.Sequential()
model.add(keras.layers.Dense(50, input_dim=1, activation='sigmoid'))
model.add(keras.layers.Dense(30, activation='sigmoid'))
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.mean_squared_error)

model.fit(X, y, epochs=50)
# %%
plt.scatter(X, y)
predictions = model.predict(X)
plt.plot(X, predictions, 'ro')
# %%
