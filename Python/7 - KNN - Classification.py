#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:59:36 2019

@author: batuhan
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("knn-dataset.csv")

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
y = data.diagnosis.values
x = data.drop(["diagnosis"],axis=1)

x = ((x-np.min(x))/(np.max(x)-np.min(x))).values

y = [1 if each =='M' else 0 for each in y]

plt.scatter(M.radius_mean,M.texture_mean,color='purple',alpha=0.4,label='kotu')
plt.scatter(B.radius_mean,B.texture_mean,color='green',alpha=0.4,label='iyi')
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()

from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train,y_train)

print("My Model's Accuracy :", knn.score(x_test,y_test))


score_list=[]
for each in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
print(score_list)

plt.subplots()
plt.plot(range(1,20),score_list)
plt.show()
# K = 2 or 4 is the best 
    
    
    
    
    
    
    
    
    
    
