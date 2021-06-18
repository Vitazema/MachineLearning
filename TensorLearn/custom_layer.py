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

from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

# Import datasets
from tensorflow.keras.datasets import fashion_mnist

# Load utils and configure GPU
sys.path.append("../")
from ml_utils import *
restrict_gpu_mem(7000)
