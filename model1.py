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
            batch_samples = samples.iloc[offset:offset+batch_size, :]
            
            image_names = []
            measurements = []
            
            for x in range(1, len(batch_samples)):
                image = batch_samples.get_value(x, 0)
                measurement = batch_samples.get_value(x, 3)
                image_names.append('data/'+image)
                measurements.append(measurement)
                
            # Load all images
            images = []
            for i in image_names:
                image = mpimg.imread(i)
                images.append(image)
            
            augmented_images=[]
            augmented_measurements=[]

            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(float(measurement)*(-1.0))

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)



driving_data = pd.read_csv('data/driving_log.csv', sep=',', header = None)

# Set our batch size
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
model.add(Cropping2D(cropping=((75,25),(0,0))))

model.add(Conv2D(6,(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
          
          
model.add(Conv2D(16,(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
          
model.add(Conv2D(8,(3,3)))
model.add(Activation('relu'))         


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.5))          
model.add(Dense(84))
model.add(Activation('relu'))  
model.add(Dropout(0.5))          
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
    train_generator,
    steps_per_epoch=ceil(len(train_samples) / batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples) / batch_size),
    epochs=10,
    verbose=1,
    )
          
#history_object = model.fit_generator(train_generator, samples_per_epoch =
 #   len(train_samples), validation_data = 
  #  validation_generator,
    #nb_val_samples = len(validation_samples), 
   # nb_epoch=1, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

model.save('model1.h5')

