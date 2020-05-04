#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 00:04:05 2020
@author: batuhan
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("clustering.csv")

from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data,method="ward")
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("Euclidian Distance")
plt.show()
#%%
from sklearn.cluster import AgglomerativeClustering

km = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage="ward")

cluster=km.fit_predict(data)
data["class"] = cluster

c = data.groupby("class")

a1 = c.get_group(0)
b1 = c.get_group(1)
c1 = c.get_group(2)

plt.figure()
plt.scatter(a1.x,a1.y)
plt.scatter(b1.x,b1.y)
plt.scatter(c1.x,c1.y)
plt.show()