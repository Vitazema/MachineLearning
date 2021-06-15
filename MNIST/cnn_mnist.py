#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#%%
import random

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Prepare environment
import sys
from tensorflow.python.keras import callbacks

from tensorflow.python.keras.engine.training_utils import handle_partial_sample_weights
sys.path.append("../")
from ml_utils import *
restrict_gpu_mem(8000)

# Load in the MNIST data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)


# %%
# Ensure the input data is accurate in shape
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not euqal to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not euqal to the number of labels"
assert(X_train.shape[1:] == (28, 28)), "The dimensions iof the images is not 28x28"
assert(X_test.shape[1:] == (28, 28)), "The dimensions iof the images is not 28x28"
print("All images data is OK")

# %%
# Plot a grid of number-images
num_of_samples = []
cols = 5
num_classes = 10

fig, ax = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 10))
fig.tight_layout()
for i in range(cols):
  for j in range(num_classes):
    x_selected = X_train[y_train == j]
    ax[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap('gray'))
    ax[j][i].axis('off')
    if i == 2:
      ax[j][i].set_title(str(j))
      num_of_samples.append(len(x_selected))
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Number distribution")
plt.xlabel('Class number')
plt.ylabel('Number of images')
# %%

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

X_train = X_train / 255.
X_test = X_test / 255.
# %%
# Define LeNet model function
def leNet_model():
  model = Sequential()
  model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2))) # Cuts the dimention by half
  model.add(Conv2D(15, (3, 3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5)) # Often placed 1 between layers, haved many parameters to prevent overfitting. More than 1 can be used for desired perfomance
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(Adam(0.01), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
  return model

def generate_model():
  model = Sequential([
    # First convolutional layer
    Conv2D(32, 3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    # Second convolutional layer
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    # Fully connected classifier
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(10, activation='softmax')
  ])
  model.compile(Adam(0.01), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
  return model

model = generate_model()
#%%
model.summary()
# %%
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.1, callbacks=[create_tensorboard_callback("-general")])

# %%
score = model.evaluate(X_test, y_test, verbose=0)
print(type(score))
print('Test score ', score[0])
print('Test accuracy ', score[1])

# %%
# Check-in handwritten image number
#img = plt.imread('./handwritten_images/Untitled.png')
# %%
from PIL import Image
with tf.keras.preprocessing.image.load_img("./handwritten_images/Untitled.png") as img:
  plt.imshow(img)
  img = img.convert('L') # Convert color image to gray
  input_arr = tf.keras.preprocessing.image.img_to_array(img)
  print(input_arr.shape)


#input_arr = np.array([input_arr])  # Convert single image to a batch.
input_arr = input_arr / 255 # Normalize image

input_arr = input_arr.reshape(1, 28, 28, 1)
#%%

prediciton = model.predict(input_arr)
print(np.argmax(prediciton, axis=-1))
# %%