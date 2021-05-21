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
    
    # num_itts = 10

    # costs = [];
    # approx_bic = []
    
    # for i in range(100):
    #     c = 0
    #     a_b = 0
    #     for r in range(num_itts):
    #         kmeans = KMeans(n_clusters=i+1, random_state=r).fit(train)

    #         k = numpy.shape(kmeans.cluster_centers_)[0]*(numpy.shape(kmeans.cluster_centers_)[1] + 1)
    #         m = len(train)
        
    #         c += kmeans.inertia_
    #         a_b += m*numpy.log(kmeans.inertia_ / m) + k*numpy.log(m)

    #     costs.append(c / num_itts)
    #     approx_bic.append(a_b / num_itts)
        
    # fig = plt.figure(figsize=[20, 6])
    # ax = fig.add_subplot(1, 2, 1)
    # ax.plot(costs)
    # ax.set_xlabel('Number of Clusters')
    # ax.set_ylabel('Reconstruction Cost');  

    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(approx_bic)
    # ax.set_xlabel('Number of Clusters')
    # ax.set_ylabel('Approximate BIC');  
    
    kmeans = KMeans(n_clusters=6, random_state=None).fit(train)
    
    
    
    
    
    
    
    
    