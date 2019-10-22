# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:39:59 2019

@author: B2HAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('pokemon.csv')

#f,ax = plt.subplots(figsize=(18,18))
#sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
data1 = data.head()

melted = pd.melt(frame=data1, id_vars='Name' , value_vars=['Attack','Defense'])