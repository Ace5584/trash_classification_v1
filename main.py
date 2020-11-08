import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import random
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

# Create Folders
# os.mkdir('reformat_data')
# # Train data
# os.mkdir('reformat_data/train')
# os.mkdir('reformat_data/train/cardboard')
# os.mkdir('reformat_data/train/glass')
# os.mkdir('reformat_data/train/metal')
# os.mkdir('reformat_data/train/paper')
# os.mkdir('reformat_data/train/plastic')
# os.mkdir('reformat_data/train/trash')
# # Test data
# os.mkdir('reformat_data/test')
# os.mkdir('reformat_data/test/cardboard')
# os.mkdir('reformat_data/test/glass')
# os.mkdir('reformat_data/test/metal')
# os.mkdir('reformat_data/test/paper')
# os.mkdir('reformat_data/test/plastic')
# os.mkdir('reformat_data/test/trash')


SOURCE_CARDBOARD_DIR = 'dataset-resized/cardboard'
SOURCE_GLASS_DIR = 'dataset-resized/glass'
SOURCE_METAL_DIR = 'dataset-resized/metal'
SOURCE_PAPER_DIR = 'dataset-resized/paper'
SOURCE_PLASTIC_DIR = 'dataset-resized/plastic'
SOURCE_TRASH_DIR = 'dataset-resized/trash'

TRAIN_DIR = 'reformat_data/train'
TEST_DIR = 'reformat_data/test'

TRAIN_CARDBOARD_DIR = 'reformat_data/train/cardboard'
TRAIN_GLASS_DIR = 'reformat_data/train/glass'
TRAIN_METAL_DIR = 'reformat_data/train/metal'
TRAIN_PAPER_DIR = 'reformat_data/train/paper'
TRAIN_PLASTIC_DIR = 'reformat_data/train/plastic'
TRAIN_TRASH_DIR = 'reformat_data/train/trash'

TEST_CARDBOARD_DIR = 'reformat_data/test/cardboard'
TEST_GLASS_DIR = 'reformat_data/test/glass'
TEST_METAL_DIR = 'reformat_data/test/metal'
TEST_PAPER_DIR = 'reformat_data/test/paper'
TEST_PLASTIC_DIR = 'reformat_data/test/plastic'
TEST_TRASH_DIR = 'reformat_data/test/trash'

# Copying File
# SPLIT_SIZE = 0.9
#
# split_data(SOURCE_CARDBOARD_DIR, TRAIN_CARDBOARD_DIR, TEST_CARDBOARD_DIR, SPLIT_SIZE)
# split_data(SOURCE_GLASS_DIR, TRAIN_GLASS_DIR, TEST_GLASS_DIR, SPLIT_SIZE)
# split_data(SOURCE_METAL_DIR, TRAIN_METAL_DIR, TEST_METAL_DIR, SPLIT_SIZE)
# split_data(SOURCE_PAPER_DIR, TRAIN_PAPER_DIR, TEST_PAPER_DIR, SPLIT_SIZE)
# split_data(SOURCE_PLASTIC_DIR, TRAIN_PLASTIC_DIR, TEST_PLASTIC_DIR, SPLIT_SIZE)
# split_data(SOURCE_TRASH_DIR, TRAIN_TRASH_DIR, TEST_TRASH_DIR, SPLIT_SIZE)

print(len(os.listdir(TRAIN_CARDBOARD_DIR)), len(os.listdir(TEST_CARDBOARD_DIR)))
print(len(os.listdir(TRAIN_GLASS_DIR)), len(os.listdir(TEST_GLASS_DIR)))
print(len(os.listdir(TRAIN_METAL_DIR)), len(os.listdir(TEST_METAL_DIR)))
print(len(os.listdir(TRAIN_PAPER_DIR)), len(os.listdir(TEST_PAPER_DIR)))
print(len(os.listdir(TRAIN_PLASTIC_DIR)), len(os.listdir(TEST_PLASTIC_DIR)))
print(len(os.listdir(TRAIN_TRASH_DIR)), len(os.listdir(TEST_TRASH_DIR)))

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(384, 512, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(6, activation='softmax'))


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(512, 384), batch_size=10, class_mode='categorical'
)


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR, target_size=(512, 384), batch_size=10, class_mode='categorical'
)

model.compile(optimizer=keras.optimizers.SGD(lr=0.001), loss='categorical_crossentropy', metrics='acc')

history = model.fit_generator(train_generator, epochs=150, verbose=1, validation_data=validation_generator)

model.save('model.h5')

