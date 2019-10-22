# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:45:43 2019

@author: B2HAN
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("linear-regression-dataset.csv",sep=";")

plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

from sklearn.linear_model import LinearRegression

linear_reg=LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

linear_reg.predict([[0]])

b0 = linear_reg.intercept_

b1 = linear_reg.coef_
