#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:19:40 2020

@author: batuhan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x1 = np.random.normal(15,5,1000)
y1 = np.random.normal(50,5,1000)

x2 = np.random.normal(60,5,1000)
y2 = np.random.normal(45,5,1000)

x3 = np.random.normal(35,5,1000)
y3 = np.random.normal(10,5,1000)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

data = pd.DataFrame({"x":x,"y":y })

np.savetxt("clustering.csv", data, delimiter=",")

plt.scatter(x,y)