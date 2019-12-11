#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:42:09 2019

@author: batuhan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv("logistic-regression.csv")

raw_data.drop(["Unnamed: 32","id"],axis=1,inplace=True)

raw_data.diagnosis=[1 if each=='M' else 0 for each in raw_data.diagnosis]

# y değerlerim sonuçlarım , x değerlerim feature'larım
y = raw_data.diagnosis.values
x = raw_data.drop(["diagnosis"],axis=1)

x = (x-np.min(x))/(np.max(x)-np.min(x)).values
#%%
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
#%%
def init(dimension):
    w = np.full((dimension,1),0.01)
    b=0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forward_backward_prop(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -(y_train*np.log(y_head)+(1-y_train)*np.log(1-y_head))
    cost = np.sum(loss)/(x_train.shape[1])
    
    weight = (np.dot(x_train,(y_head-y_train).T))/x_train.shape[1]
    bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    grad = {"weight":weight,"bias":bias}
    
    return cost,grad
#%%
def update(w,b,x_train,y_train,learning_rate,num):
    cost_list=[]
    index=[]
    for i in range(num):
        cost , grad = forward_backward_prop(w,b,x_train,y_train)
        w = w - learning_rate*grad["weight"]
        b = b - learning_rate*grad["bias"]
        if i % 10 == 0:
            cost_list.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    parameters={"weight":w,"bias":b}
    plt.plot(index,cost_list)
    plt.xlabel("Num of iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters,grad,cost_list

def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    prediction = np.zeros((1,x_test.shape[1]))
    for i in range(z.shape[1]):
        if(z[0,i]<=0.5):
            prediction[0,i]=0
        else:
            prediction[0,i]=1
    return prediction
#%%
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num):
    w,b = init(x_train.shape[0])
    parameters,grad,cost_list = update(w,b,x_train,y_train,learning_rate,num)
    
    prediction = predict(parameters["weight"],parameters["bias"],x_test)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction - y_test)) * 100))





















