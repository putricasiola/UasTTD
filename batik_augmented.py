# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:50:02 2024

@author: Eve
"""

import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Corrected import statement
from keras.preprocessing.image import ImageDataGenerator
print("TensorFlow version:", tf.__version__)

# Create an instance of ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=45,  # Random rotation between 0 and 45 degrees
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect'  # You can also try 'nearest', 'constant', 'wrap'
)

# Ensure the directory path is correct and contains subdirectories for each class
i = 0
for batch in datagen.flow_from_directory(
        directory='TTD_A/',  # Correct the path as needed
        target_size=(64, 64),
        batch_size=15,
        color_mode='rgb',
        save_to_dir='Augmented_TTD/TTD_BRIANT_CJ',  # Ensure this directory exists
        save_prefix='aug',
        save_format='png'):
    i += 1
    if i > 100:  # Limit the number of batches to augment
        break