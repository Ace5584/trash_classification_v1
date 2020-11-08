import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import random
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import transform
import pygame

item_list = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

TRAIN_DIR = 'reformat_data/train'
TEST_DIR = 'reformat_data/test'

model = keras.models.load_model('model.h5')


def process_result(li):
  temp = list()
  for item in li:
    for num in item:
      temp.append(num)
  li = temp.copy()
  highest_val = max(li)
  index = li.index(highest_val)
  return item_list[index]

def load(filename):
  np_image = Image.open(filename)
  np_image = np.array(np_image).astype('float32') / 255
  np_image = transform.resize(np_image, (512, 384, 3))
  np_image = np.expand_dims(np_image, axis=0)
  return np_image

image = load('dataset-resized/metal/metal10.jpg')

result = model.predict(image)
result = process_result(result)
print(result)
# converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
# tflite_model = converter.convert()
#
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)

# C:\Users\Alex Lai.DESKTOP-AJOHRHM\Desktop\Trash_sorting\dataset-resized\cardboard



# model.predict_classes()

