# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:55:44 2021

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

#import linear regressor model from sk learn 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


if __name__=="__main__":
    
    #extract data from csv 
    data = pd.read_csv('CAB420_Assessment_1A_Data\Data\Q1\communities.csv')
    print(data.shape)
    print("Data cleaning ---------------------------")
    #clean data 
    occurrences = np.count_nonzero(data == '?',axis = 0)
    print(occurrences)
    print(occurrences.shape)
    #remove first 5 columns 
    #print(data.columns)  
    data.drop([" state ", " county "," community "," communityname string"," fold "], axis = 1, inplace = True)
    #print(data)
    #print(data.columns)  
    #remove columns with missing data
        #most sample didnt have values 1675 out of 1994
    data.drop([" PolicBudgPerPop ", " LemasGangUnitDeploy "," LemasPctPolicOnPatr "," PolicOperBudg "," PolicCars "], axis = 1, inplace = True) 
    data.drop([data.columns[112],data.columns[111],data.columns[110],data.columns[109],data.columns[108],data.columns[107],data.columns[106],data.columns[105],data.columns[104],data.columns[103],data.columns[102],data.columns[101],data.columns[100],data.columns[99],data.columns[98],data.columns[97],data.columns[96]], axis = 1, inplace = True) 
    #print(data)
    #print(data.columns)  
    #remove row with missing value 

    # Get indexes where name column has value ?
    indexNames = data[data[' OtherPerCap '] == '?'].index
    #print(indexNames)
    # Delete these row indexes from dataFrame
    data.drop(indexNames , inplace=True)

    
    #check for no values 
    occurrences = np.count_nonzero(data == '?',axis = 0)
    #print(occurrences)
    #print(occurrences.shape)
    #standardization of inputs
    pass # todo 
    #get X and Y 
    X = data.iloc[:, :100]
    Y = data.iloc[:, 100:]
    #print(X.columns)
    #print(Y.columns)
    #print(X.shape)
    
    #print(X)
    
    #Standardised Data
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X) 
    
    #print(X)
    
    #shuffle data
    X, Y = shuffle(X, Y, random_state=None)
    
    #splite data into testing/validation/training
    X_train, X_remaining, Y_train, Y_remaining = train_test_split(X,Y, train_size = .6, random_state = 0)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_remaining,Y_remaining, train_size = .5, random_state = 0)
     
    #conver dp to numpy array 
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    X_valid = X_valid.to_numpy()
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()
    Y_valid = Y_valid.to_numpy()
    
  
    #create linear regressor 
    print("linear regressor ---------------------------")
    
    clf = GridSearchCV(LinearRegression(),{
            'copy_X' : [True,False],
            'normalize' : [True,False],
            'fit_intercept' : [True,False],
            },cv = 4,return_train_score = False)
    clf.fit(X_valid,Y_valid)
    #print(clf.cv_results_)
    print(clf.best_score_)
    print(clf.best_params_)
    
    
    
    LR = LinearRegression(copy_X= True, fit_intercept= False, normalize= True)
    #{'copy_X': True, 'fit_intercept': False, 'n_jobs': None, 'normalize': True}
    LR.fit(X_train,Y_train)    
    
    pred = LR.predict(X_test)
    
    #get R2, adj R2,mean absoulte error 
    
    # The coefficients
    print('Coefficients: \n', LR.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'% mean_squared_error(Y_test, pred))
    # The coefficient of determination: 1 is perfect prediction
    print('R2: %.2f' % r2_score(Y_test, pred))

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    #plt.title('Linear Regression Model - T1v1 - Trained on 50%')
    #plt.xlabel("Samples")
    #plt.ylabel("Resistance (KN) ")
    plt.xlim((100,120))
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(pred)
    plt.legend(['Train','Predictions'],loc='lower right')
    #plt.title('Linear Regression Model - T1v1 - Trained on 50%')
    #plt.xlabel("Samples")
    #plt.ylabel("Resistance (KN) ")
    plt.show()
    
    
    
    print("Lasso---------------------------------")
    
    clf = GridSearchCV(linear_model.Lasso(copy_X= True, fit_intercept= False, normalize= True),{
            'alpha' : [.0001,.0002,.0004,.0006,.0008,.002,.004,.006,.008,.01,.02,.04],
            },cv = 4,return_train_score = False)
    clf.fit(X_valid,Y_valid)
    
    
    #print(clf.cv_results_)
    print(clf.best_score_)
    print(clf.best_params_)
    
    #create lasso regression model
    LR_lasso = linear_model.Lasso(alpha=.002,copy_X= True, fit_intercept= False, normalize= True)
    LR_lasso.fit(X_train,Y_train) 
    pred_lasso = LR_lasso.predict(X_test)
    
    #get R2, adj R2,mean absoulte error 
    
    # The coefficients
    print('Coefficients: \n', LR_lasso.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'% mean_squared_error(Y_test, pred_lasso))
    # The coefficient of determination: 1 is perfect prediction
    print('R2: %.2f' % r2_score(Y_test, pred_lasso))

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(pred_lasso)
    plt.legend(['Train','Predictions'],loc='lower right')
    #plt.title('Linear Regression Model - T1v1 - Trained on 50%')
    #plt.xlabel("Samples")
    #plt.ylabel("Resistance (KN) ")
    plt.xlim((100,120))
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(pred_lasso)
    plt.legend(['Train','Predictions'],loc='lower right')
    #plt.title('Linear Regression Model - T1v1 - Trained on 50%')
    #plt.xlabel("Samples")
    #plt.ylabel("Resistance (KN) ")
    plt.show()
    
    print("Ridge---------------------------")
    
    clf = GridSearchCV(linear_model.Ridge(copy_X= True, fit_intercept= False, normalize= True),{
            'alpha' : [1,5,10,15,20,25,30,35,40,45,50,55,60],
            },cv = 4,return_train_score = False)
    clf.fit(X_valid,Y_valid)
    
    
    #print(clf.cv_results_)
    print(clf.best_score_)
    print(clf.best_params_)
    
    #create ridge regression model
    LR_Ridge = linear_model.Ridge(alpha=15)
    LR_Ridge.fit(X_train,Y_train) 
    pred_Ridge = LR_Ridge.predict(X_test)
    
    #get R2, adj R2,mean absoulte error 
    
    # The coefficients
    print('Coefficients: \n', LR_Ridge.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'% mean_squared_error(Y_test, pred_Ridge))
    # The coefficient of determination: 1 is perfect prediction
    print('R2: %.2f' % r2_score(Y_test, pred_Ridge))

    
    #visualise
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(pred_Ridge)
    plt.legend(['Train','Predictions'],loc='lower right')
    #plt.title('Linear Regression Model - T1v1 - Trained on 50%')
    #plt.xlabel("Samples")
    #plt.ylabel("Resistance (KN) ")
    plt.xlim((100,120))
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(Y_test)
    plt.plot(pred_Ridge)
    plt.legend(['Train','Predictions'],loc='lower right')
    #plt.title('Linear Regression Model - T1v1 - Trained on 50%')
    #plt.xlabel("Samples")
    #plt.ylabel("Resistance (KN) ")
    plt.show()
    
    #select lambda
   
    #
    
    pass 

    