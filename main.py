# SETUP
from __future__ import absolute_import, division, print_function, unicode_literals
from base64 import b64decode, b64encode

import matplotlib.pylab as plt
import tensorflow as tf
# import tensorflow_hub as hub
import json

from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image

DATA_PATH = 'batch-1'

#MODEL

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.preprocessing import image
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from os import path
from pathlib import Path
# from IPython.display import SVG, display, Image

new_model = tf.keras.models.load_model('saved_model')

# Check its architecture
new_model.summary()

BATCH_SIZE = 32
MODEL_INCEPTION_V3 = {
    "shape": (299, 299),
    "url": "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
    "preprocessor": inception_v3_preprocess_input
}
MODEL_MOBILENET_V2 = {
    "shape": (224, 224),
    "url": "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    "preprocessor": mobilenet_preprocess_input
}

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=BATCH_SIZE, arch=MODEL_MOBILENET_V2):
        'Initialization'
        self.batch_size = batch_size
        self.arch = arch

    def __getitem__(self, img):
      X = np.zeros((1, *self.arch["shape"], 3))

      pil_image = Image.open(img)
      # Image.NEAREST (0), Image.LANCZOS (1), Image.BILINEAR (2), Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)
      resized = pil_image.convert('RGB').resize(self.arch["shape"], Image.NEAREST)

      img_array = image.img_to_array(resized)
      img_array = self.arch["preprocessor"](img_array)
      X[0,] = img_array
      return X

    # def __getitem__(self, img_path):
      # X = np.zeros((1, *self.arch["shape"], 3))

      # loaded_image = image.load_img(img_path, target_size=self.arch["shape"])

      # image_array = image.img_to_array(loaded_image)
      # image_array = self.arch["preprocessor"](image_array)
      # X[0,] = image_array

      return X

#
piece_lookup = {
    0 : "K",
    1 : "Q",
    2 : "R",
    3 : "B",
    4 : "N",
    5 : "P",
    6 : "k",
    7 : "q",
    8 : "r",
    9 : "b",
    10 : "n",
    11 : "p",
    12 : "1",
}
def y_to_fens(y):
  fen = ""
  for sq in range(64):
    piece_idx = np.argmax(y[sq][0,])
    fen += piece_lookup[piece_idx]
  a = [fen[i:i+8] for i in range(0, len(fen), 8)]
  a = a[::-1]
  fen = "/".join(a)
  for i in range(8,1,-1):
    old_str = "1" * i
    new_str = str(i)
    fen = fen.replace(old_str, new_str)
  return fen



#### TEST ####

validation_generator = DataGenerator()

# label = "batch-1\\RRrR2NN-1K1qnkBp-N1kqpnRr-q2qKNb1-n3B1Q1-N1PkNN1K-r2r2n1-P3K1Kq.jpg"

# processed_image2 = validation_generator.__getitem2__(label)
# prediction2 = new_model.predict(processed_image2)

# prediction_fen2 = y_to_fens(prediction2)
# print(prediction_fen2)




############# flask app ###############

from flask import Flask
from flask import request, jsonify

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def new_user():
    image_data = request.files['image']

    # TODO refactor
    processed_image = validation_generator.__getitem__(image_data)
    prediction = new_model.predict(processed_image)

    prediction_fen = y_to_fens(prediction)
    print(prediction_fen)

    res = {"prediction": prediction_fen}
    return jsonify(res)