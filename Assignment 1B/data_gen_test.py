# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:11:50 2021

@author: User
"""

import scipy.io
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import numpy
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

if __name__=="__main__":
   
    #import data
    test_data = scipy.io.loadmat('CAB420_Assessment_1B_Data\Data\Q1\q1_test.mat')
    train_data = scipy.io.loadmat('CAB420_Assessment_1B_Data\Data\Q1\q1_train.mat')
   
    # Load images and labels
    test_Y = np.array(test_data['test_Y'])
    test_X = np.array(test_data['test_X'])  /255.0
   
    train_Y = np.array(train_data['train_Y'])
    train_X = np.array(train_data['train_X']) /255.0
   
    # Check the shape of the data
    print(test_X.shape)
    print(train_X.shape)
   
    # Fix the axes of the images
    test_X = np.moveaxis(test_X, -1, 0)
    train_X = np.moveaxis(train_X, -1, 0)

    print(test_X.shape)
    print(train_X.shape)
   
    # Plot a random image and its label

    plt.imshow(train_X[590])
    plt.show()
    print(train_Y[590])
    
    plt.imshow(test_X[590])
    plt.show()
    print(test_Y[590])
    print("end of printing images--------")
   
    #reshape train Y to vector format
    print(test_Y)
   
    #replace 10 to 0s in ys
    train_Y = np.where(train_Y==10, 0, train_Y)
    test_Y = np.where(test_Y==10, 0, test_Y)
      
    a = 5

    print(test_Y[a])
    print(test_Y[a+1])
   
    def unique(list1):
        x = np.array(list1)
        print(np.unique(x))
       
    print("unique")
           
    unique(test_Y)
    
    gen = ImageDataGenerator(
        #rotation_range=45,
        #width_shift_range=0.1,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #rescale=1./255,
        #channel_shift_range=10,
        #fill_mode='nearest'
        )
    
    aug_iter = gen.flow(train_X[1])
    
    aug_img = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
    
    for i in range(10):
        
        plt.imshow(aug_img[i])
        plt.show()
    