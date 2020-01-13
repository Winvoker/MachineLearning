#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:35:59 2019

@author: batuhan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("knn-dataset.csv")

data.drop(["id", "Unnamed: 32"],axis=1,inplace=True)

data.diagnosis = [1 if each=='M' else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1) 

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=41,test_size=0.3)

from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train,y_train)
print("Accuracy : ",svm.score(x_test,y_test))
