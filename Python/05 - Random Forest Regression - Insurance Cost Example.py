#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:25:30 2020

@author: batuhan
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
data.sex = [1 if each == "male" else 0 for each in data.sex]
data.smoker = [1 if each == "yes" else 0 for each in data.smoker]

data = pd.get_dummies(data)

gr = data.groupby("smoker")
data1 = gr.get_group(1) # smoker
data2 = gr.get_group(0) # non-smoker



y = data.charges.values.reshape(-1,1)
x = data.drop(["charges"],axis=1)

v1, a1 = data1.age , data2.age
v2, a2 = data1.charges, data2.charges

plt.figure(1)
plt.scatter(v1,v2,color="r",alpha=0.5,label="smoker")
plt.scatter(a1,a2,color="g",alpha=0.5,label = "non-smoker")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.15,random_state=1)


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 100,random_state=1)
forest.fit(x_train,y_train)
print("acc: ",forest.score(x_test,y_test))

plt.figure(2)
pred = forest.predict(x_test).reshape(-1,1)
plt.scatter(pred,pred-y_test,alpha=0.5)
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'green')
plt.show()
