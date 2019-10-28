# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:45:43 2019

@author: B2HAN
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("linear-regression-dataset.csv",sep=";")
# plotting data 
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

# implementing linear regression
linear_reg=LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#predicting when deneyim = 10 
linear_reg.predict([[10]])

# y = b0 + b1.x 
b0 = linear_reg.intercept_

b1 = linear_reg.coef_

# plotting our fit line
y_predicted = linear_reg.predict(x)

plt.plot(x,y_predicted,color='r')
plt.show()
