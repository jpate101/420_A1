# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:15:35 2021

@author: User
"""

import numpy as np
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import scipy.io


if __name__=="__main__":
    # Set random state
    np.random.seed(20)
    
    # Load images and labels
    #import data 
    test_data = scipy.io.loadmat('CAB420_Assessment_1B_Data\Data\Q1\q1_test.mat')
    train_data = scipy.io.loadmat('CAB420_Assessment_1B_Data\Data\Q1\q1_train.mat')
    
    # Load images and labels
    test_labels = np.array(test_data['test_Y'])
    test_images = np.array(test_data['test_X'])
    
    train_labels = np.array(train_data['train_Y'])
    train_images = np.array(train_data['train_X'])
    
    # Check the shape of the data
    print(train_images.shape)
    print(test_images.shape)
    # Fix the axes of the images
    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)
    print("reshaped x")
    print(train_images.shape)
    print(test_images.shape)
    
    # Plot a random image and its label
    plt.imshow(train_images[427])
    plt.show()
    print("random img")
    print(train_labels[427])
    
    # Convert train and test images into 'float64' type
    train_images = train_images.astype('float64')
    test_images = test_images.astype('float64')
    # Convert train and test labels into 'int64' type
    train_labels = train_labels.astype('int64')
    test_labels = test_labels.astype('int64')
    
    #insert data normalisation 
    
    # One-hot encoding of train and test labels
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)
    print("LabelBinarizer")
    print(train_labels[427])
    
    # Define actual model

    keras.backend.clear_session()

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                           activation='relu',
                           input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                        activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
    
        keras.layers.Conv2D(64, (3, 3), padding='same', 
                           activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
    
        keras.layers.Conv2D(128, (3, 3), padding='same', 
                        activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
    
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),    
        keras.layers.Dense(10,  activation='softmax')
    ])


    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    model.summary()
    
    # Fit model in order to make predictions

    history = model.fit(train_images, train_labels, batch_size=1,epochs=1)
    
    #predictions = model.predict(test_X) 
    predictions = model.predict(test_images) 
    
    a = 450
    
    print(predictions[a])
    print(test_labels[a])

    print(predictions[a+1])
    print(test_labels[a+1])

    print(predictions[a+2])
    print(test_labels[a+2])

    print(predictions[a+30])
    print(test_labels[a+30])
    
    
    pass