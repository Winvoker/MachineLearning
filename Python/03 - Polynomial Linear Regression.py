# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:40:46 2019

@author: B2HAN
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


df = pd.read_csv("polynomial-regression.csv",sep=";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)


polynomial_reg = PolynomialFeatures(degree=8)

x_polynomial = polynomial_reg.fit_transform(x)

linear_reg=LinearRegression()

linear_reg.fit(x_polynomial,y)

y_head2 = linear_reg.predict(x_polynomial)

plt.plot(x,y_head2,color="green")
plt.show()
