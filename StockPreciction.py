import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import quandl
quandl.ApiConfig.api_key = "g5x4nVyzgx-hKs6s7Nt2"
import pandas_datareader as pdr
import datetime

# Make call to the yahoo API to get the desired data.
aapl = pdr.get_data_yahoo('AAL',  start=datetime.datetime(1990, 10, 1), end=datetime.datetime(2018, 5, 8))

# We can do the same with Quandl.
data = quandl.get_table('WIKI/PRICES', qopts = {'columns': ['close', 'volume']}, ticker = ['AAPL'], date = { 'gte': '2016-01-01', 'lte': '2018-05-01' }, paginate=True)

# Extract the close of the desired stock.
stockClose = aapl.iloc[:, [3]].values

# Normalize the data so that it has the same noise of the model when we will had a few other variables into the mix.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(stockClose)

# Setting the training and testing length accordingly.
train_size = int(len(aapl) * 0.80)
test_size = len(aapl) - train_size
train_data = training_set_scaled[0:train_size]
test_data = training_set_scaled[:-train_size]

# Period length
period = 3

# Two arrays that will hold the referrence to the price and wheter it has increased in a predetermined time frame.
prices = []
goodBuy = []

# Set a new array and make the array so that if a price is larger in 30 trading days it result in a 1 or a 0 is not.
for i in range(period, train_size):
    prices.append(train_data[i - period:i, 0])
    goodBuy.append(train_data[i, 0])
prices, goodBuy = np.array(prices), np.array(goodBuy)

prices = np.reshape(prices, (prices.shape[0], prices.shape[1], 1))

# Define the model.
model = Sequential()

# Add layers of node to the model.
model.add(LSTM(units=50, return_sequences=True, input_shape=(prices.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(prices, goodBuy, epochs =5, batch_size = 15)


# Separate the code of the array that will constitute the test data, reshape and normalize it.
inputs = stockClose[train_size:]
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

# Empty array that will be filled with the past period of data used for the prediction which will be used for the prediction.
X_test = []
for i in range(period, test_size):
    X_test.append(inputs[i - period:i, 0])


X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Predict the prices using the model and do the inverse of the normalization done.
predicted_stock_price = model.predict(X_test)
predictedNormalPrice = scaler.inverse_transform(predicted_stock_price)


# Calculate the buffer of the second array that need to be used for the real stock prices.
buffer = train_size + period

# Setup the display of the graphing.
plt.plot(stockClose[buffer:], color = 'black', label = 'AAL Stock Price')
plt.plot(predictedNormalPrice, color = 'red', label = 'Predicted AAL Stock Price')
plt.title('Deep Learning Prediction')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price in US($)')
plt.legend()
plt.show()





