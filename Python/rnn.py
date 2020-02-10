# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:56:39 2020

@author: B2HAN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# RNN

data_train = pd.read_csv("Stock_Price_Train.csv")
data_test = pd.read_csv("Stock_Price_Test.csv")

train = data_train.Open.values.reshape(-1,1)
test = data_test.Open.values.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)

#plt.plot(train)
#plt.show()

x_train, y_train=[],[]

timesteps = 50
for i in range(timesteps, 1258):
    x_train.append(train[i-timesteps:i,0])
    y_train.append(train[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#%%
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.models import Sequential

model = Sequential()

model.add(SimpleRNN(units=50, activation='tanh',return_sequences = True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.3))
model.add(SimpleRNN(units=50, activation="tanh", return_sequences = True ))
model.add(Dropout(0.3))
model.add(SimpleRNN(units=50, activation="tanh", return_sequences = True ))
model.add(Dropout(0.3))
model.add(SimpleRNN(units=50))
model.add(Dropout(0.3))

model.add(Dense(units = 1))

model.compile(optimizer='adam', loss="mean_squared_error")
model.fit(x_train, y_train, epochs=100, batch_size=50)

#%%

dataset_total = pd.concat((data_train["Open"], data_test["Open"]), axis = 0)

inputs = dataset_total[len(dataset_total) - len(test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(timesteps, 70):
    x_test.append(inputs[i-timesteps:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = model.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
#%%
plt.plot(test, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()