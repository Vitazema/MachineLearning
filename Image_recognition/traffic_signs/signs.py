#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import pickle
import random

import sys

from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.metrics import CategoricalCrossentropy
sys.path.append("../../")
from ml_utils import *
restrict_gpu_mem(7000)

# %%
with open('./train.p', 'rb') as f:
  train_data = pickle.load(f)
with open('./valid.p', 'rb') as f:
  val_data = pickle.load(f)
with open('./test.p', 'rb') as f:
  test_data = pickle.load(f)

print(type(train_data))
X_train, y_train =  train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# %%
print(X_train.shape, X_val.shape, X_test.shape)

# %%
# Ensure data is correct 
assert(X_train.shape[0] == y_train.shape[0]), 'The number of images is not equal to the number of labels'
assert(X_train.shape[1:] == (32, 32, 3)), "The dimentions of the images are not 32x32x3"
df = pd.read_csv('./signnames.csv')

# %%
num_of_samples = []
cols = 5
num_classes = 43

fig, ax = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
fig.tight_layout()

for i in range(cols):
  for j, row in df.iterrows(): # (index, Series)
    x_selected = X_train[y_train == j]
    ax[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
    ax[j][i].axis('off')
    if i == 2:
      ax[j][i].set_title(str(j) + '-' + row["SignName"])
      num_of_samples.append(len(x_selected))

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# %%
import cv2
plt.imshow(X_train[1000])
print(X_train[1000].shape)
print(y_train[1000])

# %%
def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

img = grayscale(X_train[1000])
plt.imshow(img)
plt.axis('off')
print(img.shape)

# %%
# Equalization (simply increase image contrast)
def equalize(img):
  img = cv2.equalizeHist(img)
  return img

img = equalize(img)
plt.imshow(img)
plt.axis('off')
print(img.shape)

# %%
def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  img = img / 255.0 # normilization
  return img 

# %%
X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

# %%
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])

# %%
X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)

#%%
datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                zoom_range=0.2,
                                                shear_range=0.1,
                                                rotation_range=15)
datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20) # Starting the process flowing of augmented data from the generator in batches of 20
X_batch, y_batch = next(batches)

fig, ax = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
  ax[i].imshow(X_batch[i].reshape(32, 32))
  ax[i].axis('off')

# %%
# Encode categorical values
y_train = tf.keras.utils.to_categorical(y_train, 43)
y_test = tf.keras.utils.to_categorical(y_test, 43)
y_val = tf.keras.utils.to_categorical(y_val, 43)

#%%
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# %%
# Define LeNet model function
def leNet_model():
  model = tf.keras.Sequential()

  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2))) # Cut dimentions in half

  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2))) # Cut dimentions in half again
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = leNet_model()
print(model.summary())
# %%
# Fit the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), batch_size=32, epochs=10,
  validation_data=(X_val, y_val), callbacks=[create_tensorboard_callback('geenrator')])
#model.fit(X_train, y_train, validation_data=(X_val, y_val),
#batch_size=32, epochs=10, callbacks=[create_tensorboard_callback('adam-conv-drop')])



# %%
# Try on custom data
url = "https://autotonkosti.ru/sites/default/files/inline/images/2018-05-10_225247.jpg"
img = tf.keras.utils.get_file(origin=url, fname='70')
img = tf.keras.preprocessing.image.load_img(img, color_mode='grayscale', target_size=(32, 32))
plt.imshow(img, cmap=plt.get_cmap('gray'))
#%%
img = np.array(img)
img = equalize(img)
img = img / 255.0 # normilization
img = img.reshape(1, 32, 32, 1)

# %%
str(np.argmax(model.predict(img), axis=-1))
# %%
