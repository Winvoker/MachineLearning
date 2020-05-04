#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:32:26 2020

@author: batuhan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("clustering.csv",sep=",")

from sklearn.cluster import KMeans
wcss = [] # within cluster sum of squares

for i in range(1,15):
    k = KMeans(n_clusters=i)
    k.fit(data) # 
    wcss.append(k.inertia_)

plt.figure()
plt.plot(range(1,15),wcss)
plt.show() # -- I see that elbow is at 3 , therefore n_cluster = 3

km = KMeans(n_clusters=3)
clusters = km.fit_predict(data)

data["class"] = clusters

c = data.groupby("class")

a1 = c.get_group(0)
b1 = c.get_group(1)
c1 = c.get_group(2)

plt.figure()
plt.scatter(a1.x,a1.y)
plt.scatter(b1.x,b1.y)
plt.scatter(c1.x,c1.y)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="yellow")

plt.show()