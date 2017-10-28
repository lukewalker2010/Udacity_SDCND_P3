#Luke Walker
#SDCND P3 Behavioral Cloning

#Imports
import cv2
import csv
import numpy as np
import os
import sklearn

#Model Imports
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

#%%
#Function Defintions

#Get Lines
def getLines(dataPath):
    """
    Returns each line from driving log
    """
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)
    return lines

#Get Images
def getImages(dataPath):
    """
    Returns center left and right images
    """
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerAll = []
    leftAll = []
    rightAll = []
    measurementAll = []
    for directory in dataDirectories:
        lines = getLines(directory)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerAll.extend(center)
        leftAll.extend(left)
        rightAll.extend(right)
        measurementAll.extend(measurements)

    return (centerAll, leftAll, rightAll, measurementAll)

#Put images into single array
def extendImages(center, left, right, measurement, offset):
    """
    Extends images to single array
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + offset for x in measurement])
    measurements.extend([x - offset for x in measurement])
    return (imagePaths, measurements)

#Generator
def generator(samples, batch_size=32):
    """
    Generate the required images and angles for training
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator always runs
        samples = sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []

            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                #Flip and append
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            #trim images
            inputs = np.array(images)
            outputs = np.array(angles)

            #Yield and shuffle
            yield sklearn.utils.shuffle(inputs, outputs)

#%%
#Model definition
def modified_nVidia():
    """
    Creates modified nVidia model
    """
    #Sequential Model
    model = Sequential()
    #Lambda Layer
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    #Cropping Layer
    model.add(Cropping2D(cropping=((50,20), (0,0)))) #remove top 50 pix and bottom 20 pix
    #5 Convolution Layers
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    #Flatten
    model.add(Flatten())
    #4 Dense Layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    #Return Model
    return model

#%%
# Reading images locations.
centerPaths, leftPaths, rightPaths, measurements = getImages('data')
imagePaths, measurements = extendImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
print('Total Images: {}'.format(len(imagePaths)))

#Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Show number of training and validation samples
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Build the model
model = modified_nVidia()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
run_model = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model_nnVidia_modified.h5')
print(run_model.history.keys())
print('Loss')
print(run_model.history['loss'])
print('Validation Loss')
print(run_model.history['val_loss'])
