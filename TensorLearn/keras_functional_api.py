#%%
import tensorflow as tf
import pydot as pydot
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#%%
# Functional style model definition
input_layer = Input(shape=(28, 28))

# Stack the layers using the syntax: new_Layer()(previous_layer)
flatten_layer = Flatten()(input_layer)
first_dense = Dense(128, activation=tf.nn.relu)(flatten_layer)
output = Dense(10, activation=tf.nn.softmax)(first_dense)

func_model = tf.keras.models.Model(inputs=input_layer, outputs=output)
# %%
# Visualize the model graph
tf.keras.utils.plot_model(func_model, show_shapes=True,
                                    show_layer_names=True)
# %%
func_model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# %%
def format_output(data):
    y1 = data.pop('Y1')
    y1 = np.array(y1)
    y2 = data.pop('Y2')
    y2 = np.array(y2)
    return y1, y2


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()

# Specify data URI
URI = './ENB2012_data.xlsx'

# Use pandas excel reader
df = pd.read_excel(URI)
df = df.sample(frac=1).reset_index(drop=True)

# Split the data into train and test with 80 train / 20 test
train, test = train_test_split(df, test_size=0.2)
train_stats = train.describe()

# Get Y1 and Y2 as the 2 outputs and format them as np arrays
train_stats.pop('Y1')
train_stats.pop('Y2')
train_stats = train_stats.transpose()
train_Y = format_output(train)
test_Y = format_output(test)

# Normalize the training and test data
norm_train_X = norm(train)
norm_test_X = norm(test)

#%%
# Build the model

input_layer = Input(shape=(len(train .columns),))
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)

# Y1 output will be fed directly from the second dense
y1_output = Dense(units='1', name='y1_output')(second_dense)
third_dense = Dense(units='64', activation='relu')(second_dense)

# Y2 output will come via the third dense
y2_output = Dense(units='1', name='y2_output')(third_dense)

# Define the model with the input layer and a list of output layers
model = tf.keras.models.Model(inputs=input_layer, outputs=[y1_output, y2_output])

print(model.summary())

# %%
optimizer = tf.keras.optimizers.SGD(0.001)
model.compile(optimizer=optimizer)
# %%
