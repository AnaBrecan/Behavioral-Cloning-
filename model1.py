import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
import sklearn
import math
from math import ceil
from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                
                image_center = batch_sample[0]
                image_left = batch_sample[1]
                image_right = batch_sample[2]
                
                measurement = float(batch_sample[3])
                
                image_c = mpimg.imread('data/'+image_center)
                images.append(image_c)
                measurements.append(measurement)
                
                image_l = mpimg.imread('./data/IMG/' + batch_sample[1].split('/')[-1])
                images.append(image_l)
                measurements.append(measurement + 0.35)
                
                image_r = mpimg.imread('./data/IMG/' + batch_sample[2].split('/')[-1])
                images.append(image_r)
                measurements.append(measurement - 0.35)
                
                images.append(cv2.flip(image_c, 1))
                measurements.append(float(measurement) * (-1.0))

                images.append(cv2.flip(image_l, 1))
                measurements.append(float(measurement) * (-1.0) + 0.35)
                
                images.append(cv2.flip(image_r, 1))
                measurements.append(float(measurement) * -1.0 - 0.35)  


            
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)



driving_data = []
with open('data/driving_log.csv') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        driving_data.append(row)

# Set the batch size
batch_size=32

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(driving_data, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.models import Model 
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()

model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
# trim image to only see section with road
model.add(Cropping2D(cropping=((75,25),(0,0))))

model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
model.add(Conv2D(48, (3, 3), activation='relu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator,
    steps_per_epoch=ceil(len(train_samples) / batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples) / batch_size),
    epochs=2,
    verbose=1,
    )
          

model.save('model6.h5')
print("Model Saved")
model.summary()

