#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:58:39 2020

@author: batuhan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("nasa.csv")
data.columns = [c.replace(' ', '_') for c in data.columns]
data.drop(["Neo_Reference_ID","Name","Close_Approach_Date","Epoch_Date_Close_Approach"
           ,"Orbiting_Body","Orbit_Determination_Date","Equinox"],axis=1,inplace=True)
data.Hazardous = [1 if each==True else 0 for each in data.Hazardous]

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scale = MinMaxScaler()

y = data.Hazardous.values.reshape(-1,1)

x = data.drop(["Hazardous"],axis=1).values #returns a numpy array
x = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

c = data.groupby('Hazardous')
a = c.get_group(0)
b = c.get_group(1)
a1, a2 = a.Perihelion_Distance , a.Miles_per_hour
b1, b2 = b.Perihelion_Distance , b.Miles_per_hour
 
plt.scatter(a1,a2,color="b",label="Safe",alpha=0.1)
plt.scatter(b1,b2,color="r", label ="Hazardous",alpha=0.1)
plt.legend()
plt.show()


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Logistic Regression Acc : ", lr.score(x_test,y_test))

from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train,y_train)
print("SVM Acc : ",svm.score(x_test,y_test))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
print("KNN Acc:", knn.score(x_test,y_test))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
print("DCT Acc : ", dtc.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(x_train,y_train)
print("RFC : Acc : ",rfc.score(x_test,y_test))