import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("multiple-linear-regression-dataset.csv",sep=";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

mlt_linear_reg = LinearRegression()


mlt_linear_reg.fit(x, y)

mlt_linear_reg.predict([[25,50]])
