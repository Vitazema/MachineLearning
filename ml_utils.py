import tensorflow

def create_tensorboard_callback(logs_suffix=""):
  '''
  Configure training environment in current file working direcotory. 
  # Create deep multicategorial model and log to Tensorboard
    import sys
    sys.path.append("../")
    from ml_utils import create_tensorboard_callback
  Removes logs every hour
  '''
  import datetime as datetime
  import os
  import shutil
  current_dir_path = os.path.dirname(os.path.realpath(__file__))
  logdir = os.path.join(current_dir_path, "tb_logs", datetime.datetime.now().strftime("%Y%m%d-%H") + logs_suffix)
  ## Try to remove tree for new logs to not shuffle; if failed show an error using try...except on screen
  try:
      shutil.rmtree(logdir, ignore_errors=True)
  except OSError as e:
      print ("Error: %s - %s." % (e.filename, e.strerror))

  return tensorflow.keras.callbacks.TensorBoard(logdir)

def restrict_gpu_mem(memory=5000):
  '''
  Restrict GPU memory consumption
    import sys
    sys.path.append("../")
    from ml_utils import restrict_gpu_mem
  '''
  gpus = tensorflow.config.list_physical_devices('GPU')
  if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
      tensorflow.config.experimental.set_virtual_device_configuration(
          gpus[0],
          [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
      logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Virtual devices must be set before GPUs have been initialized
      print(e)


"""
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
"""