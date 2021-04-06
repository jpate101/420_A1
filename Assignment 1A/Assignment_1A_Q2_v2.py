# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:40:31 2021

@author: User
"""

# numpy handles pretty much anything that is a number/vector/matrix/array
import numpy as np
# pandas handles dataframes (exactly the same as tables in Matlab)
import pandas as pd
# matplotlib emulates Matlabs plotting functionality
import matplotlib.pyplot as plt
# stats models is a package that is going to perform the regression analysis
from statsmodels import api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
# os allows us to manipulate variables on out local machine, such as paths and environment variables
import os
# self explainatory, dates and times
from datetime import datetime, date
# a helper package to help us iterate over objects
import itertools

from sklearn.metrics import plot_confusion_matrix

#import linear regressor model from sk learn 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.utils import shuffle

#extract data from csv 
training_data = pd.read_csv('CAB420_Assessment_1A_Data/Data/Q2/training.csv')
testing_data = pd.read_csv('CAB420_Assessment_1A_Data/Data/Q2/testing.csv')
print(training_data.head)

#splite data to X and Y
Y_train = training_data.iloc[:, :1]
X_train = training_data.iloc[:, 1:]

Y_test = training_data.iloc[:, :1]
X_test = training_data.iloc[:, 1:]

#convert to numpy array (sklearn wont accept pd dataframe )

X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy().ravel()

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy().ravel()

#splite data into testing/validation
X_test, X_valid, Y_test, Y_valid = train_test_split(X_test,Y_test, train_size = 0.5, random_state = 0)

#shuffle data
X_train, Y_train = shuffle(X_train, Y_train, random_state=None)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#print(Y_test)

def eval_model(model, X_train, Y_train, X_test, Y_test):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf = plot_confusion_matrix(model, X_train, Y_train, normalize=None, xticks_rotation='vertical', ax=ax)
    pred = model.predict(X_train)
    conf.ax_.set_title('Training Set Performance: ' + str(sum(pred == Y_train)/len(Y_train)));
    ax = fig.add_subplot(1, 2, 2)
    conf = plot_confusion_matrix(model, X_test, Y_test, normalize=None, xticks_rotation='vertical', ax=ax)
    pred = model.predict(X_test)
    conf.ax_.set_title('Testing Set Performance: ' + str(sum(pred == Y_test)/len(Y_test)));
    

"""    
print("K Neighbour ---------------------------")



print("Grid search")

clf = GridSearchCV(KNeighborsClassifier(),{
        'n_neighbors' : [1,2,3,4,5,7,9,50,70],
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size' : [1,5,10,15,20,25,30,35,40],
        'p' : [1,2],
        #'n_jobs' : [1,-1],

        },cv = 4,return_train_score = False)
clf.fit(X_valid,Y_valid)
#print(clf.cv_results_)
print(clf.best_score_)
print(clf.best_params_)

training_accuracy=[]
test_accuracy=[]
neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    knn=KNeighborsClassifier(n_neighbors)
    knn.fit(X_train,Y_train)
    training_accuracy.append(knn.score(X_train,Y_train))
    test_accuracy.append(knn.score(X_test,Y_test))

print(training_accuracy,test_accuracy)

print("using real data")

#build/train K neighbour clf 
K_neigh = KNeighborsClassifier(algorithm= 'auto', leaf_size= 1, n_neighbors= 3, p= 1, weights= 'uniform')
K_neigh.fit(X_train, Y_train)

y_pred = K_neigh.predict(X_test)


#accuracy
score = accuracy_score(Y_test, y_pred)
print(score)
#visuals   
vis = confusion_matrix(Y_test, y_pred, labels=["s ", "d ", "h ","o "])
print(vis)

eval_model(K_neigh,X_train,Y_train,X_test,Y_test)

plt.plot(neighbors_settings,training_accuracy,label='Training Accuracy')
plt.plot(neighbors_settings,test_accuracy,label='Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
"""
print("Ran forest ---------------------------")


print("Grid search")

"""

clf = GridSearchCV(RandomForestClassifier(),{
        'n_estimators' : [25,50,75,100,125],
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [20,30,40,50,None],
        'min_samples_split' : [2,3,4,10],
        'min_samples_leaf' : [1,2,3,4,10],

        },cv = 4,return_train_score = False)
clf.fit(X_valid,Y_valid)
#print(clf.cv_results_)
print(clf.best_score_)
print(clf.best_params_)
"""
print("using real data")

#build/train random forest clf 
forest = RandomForestClassifier(criterion= 'gini', max_depth=40, min_samples_leaf= 2, min_samples_split= 4, n_estimators= 25)
#forest = RandomForestClassifier() 
forest.fit(X_train, Y_train)

y_pred = forest.predict(X_test)


#accuracy
score = accuracy_score(Y_test, y_pred)
print(score)
#visuals   
vis = confusion_matrix(Y_test, y_pred, labels=["s ", "d ", "h ","o "])
print(vis)

eval_model(forest,X_train,Y_train,X_test,Y_test)

"""
print("SVC ---------------------------")


print("Grid search")
clf = GridSearchCV(SVC(),{
        'C' : [1,2,3,4,5],
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'shrinking' : [True,False],
        'probability' : [True,False],
        'tol' : [1e-1,1e-2,1e-3,5e-3,9e-3,1e-4],

        },cv = 4,return_train_score = False)

#make_pipeline(StandardScaler(), clf).fit(X_valid,Y_valid)
clf.fit(X_valid,Y_valid)

#print(clf.cv_results_)
print(clf.best_score_)
print(clf.best_params_)


print("using real data")

#build/train SVM clf 
SVM = make_pipeline(StandardScaler(), SVC(C= 4, kernel= 'rbf', probability= True, shrinking= True, tol= 0.1))
SVM.fit(X_train, Y_train)

y_pred = SVM.predict(X_test)


#accuracy
score = accuracy_score(Y_test, y_pred)
print(score)
#visuals   
vis = confusion_matrix(Y_test, y_pred, labels=["s ", "d ", "h ","o "])
print(vis)

eval_model(SVM,X_train,Y_train,X_test,Y_test)
"""