# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:23:24 2021

@author: User
"""

import pandas
import numpy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from matplotlib import cm
from datetime import datetime


if __name__=="__main__":
    
    data = pandas.read_csv('data/Q1/ratings.csv');
    print(data)
    
    train = data.iloc[0:700, :]
    test = data.iloc[700:, :]
    
    print(train)
    pass