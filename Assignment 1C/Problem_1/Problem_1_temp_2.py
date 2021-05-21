# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:16:57 2021

@author: User
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
#import helper
import seaborn as sns

if __name__=="__main__":
    movies = pd.read_csv('data/Q1/movies.csv')
    print(movies.head())
    
    ratings = pd.read_csv('data/Q1/ratings.csv')
    print(ratings.head())
    
    print('The dataset contains: ', len(ratings), ' ratings of ', len(movies), ' movies.')
    
    merged_movies = pd.merge(ratings, movies,on='movieId')
    print(merged_movies.head())