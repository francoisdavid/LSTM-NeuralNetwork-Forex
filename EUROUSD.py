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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler
import math
import statistics

'''
# Make call to the yahoo API to get the desired data.
aapl = pdr.get_data_yahoo('AAL',  start=datetime.datetime(1990, 10, 1), end=datetime.datetime(2018, 5, 8))

# We can do the same with Quandl.
data = quandl.get_table('WIKI/PRICES', qopts = {'columns': ['close', 'volume']}, ticker = ['AAPL'], date = { 'gte': '2016-01-01', 'lte': '2018-05-01' }, paginate=True)

# Extract the close of the desired stock.
stockClose = aapl.iloc[:, [3]].values
'''

# DataFrame that will be filled with the information acquired from Quandl.
forexData = []
forexData = pd.DataFrame()

# Get the Forex data. Which is set to Euro to USD
forexPrices = quandl.get("SGE/BELCUR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31", end_date="2018-03-29")
forexScaler = MinMaxScaler(feature_range=(0, 1))
forexScaler.fit(forexPrices)
forexData = forexPrices
# forexData = pd.DataFrame(forexScaler.transform(forexPrices))
dataLength = len(forexPrices)


# Function used to interpolate the missing values for monthly values and others missing entries.
def interpolatedForFreq(call):
    df = pd.DataFrame(call)
    upsampled = df.resample("B").ffill()
    df = pd.DataFrame(upsampled)
    # df = df.pct_change()
    df = df.fillna(method='bfill')
    df = df.fillna(0)
    return df


# Get the Interest Rate and set them to be in the forexData dataframe.
interestRate = interpolatedForFreq(quandl.get("SGE/EURIR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31"))
forexData["InterestRate"] = interestRate
forexData = forexData.fillna(method='ffill')
interestRateScaler = MinMaxScaler(feature_range=(0, 1))
iR = np.array(forexData["InterestRate"]).reshape(-1, 1)
interestRateScaler.fit(iR)
forexData["InterestRate"] = interestRateScaler.transform(iR)

# Get the Inflation Rate and set them to be in the forexData dataframe.
inflationRate = interpolatedForFreq(
    quandl.get("SGE/EURCPIC", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31"))
forexData["InflationRate"] = inflationRate
forexData = forexData.fillna(method='ffill')
inflationRateScaler = MinMaxScaler(feature_range=(0, 1))
iF = np.array(forexData["InflationRate"]).reshape(-1, 1)
inflationRateScaler.fit(iF)
forexData["InflationRate"] = inflationRateScaler.transform(iF)

# Get the Imports and set them to be in the forexData dataframe.
imports = interpolatedForFreq(quandl.get("SGE/EURIMVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31"))
forexData['Imports'] = imports
forexData = forexData.fillna(method='ffill')
importsScaler = MinMaxScaler(feature_range=(0, 1))
imp = np.array(forexData["Imports"]).reshape(-1, 1)
importsScaler.fit(imp)
forexData["Imports"] = importsScaler.transform(imp)

# Get the Exports and set them to be in the forexData dataframe.
exports = interpolatedForFreq(quandl.get("SGE/EUREXVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31"))
forexData["Exports"] = exports
forexData = forexData.fillna(method='ffill')
exportsScaler = MinMaxScaler(feature_range=(0, 1))
exp = np.array(forexData["Exports"]).reshape(-1, 1)
exportsScaler.fit(exp)
forexData["Exports"] = exportsScaler.transform(exp)

# Get the Exports and set them to be in the forexData dataframe.
gdp = interpolatedForFreq(quandl.get("SGE/EURG", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31"))
forexData["GDP"] = gdp
forexData = forexData.fillna(method='ffill')
gdpScaler = MinMaxScaler(feature_range=(0, 1))
gdpp = np.array(forexData["GDP"]).reshape(-1, 1)
gdpScaler.fit(gdpp)
forexData["GDP"] = gdpScaler.transform(gdpp)

# Get the Consumer Spending and set them to be in the forexData dataframe.
consumerSpending = interpolatedForFreq(
    quandl.get("SGE/EURCSP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31"))
forexData["ConsumerSpending"] = consumerSpending
forexData = forexData.fillna(method='ffill')
consumerSpendingScaler = MinMaxScaler(feature_range=(0, 1))
consumerSpend = np.array(forexData["ConsumerSpending"]).reshape(-1, 1)
consumerSpendingScaler.fit(consumerSpend)
forexData["ConsumerSpending"] = gdpScaler.transform(consumerSpend)

# Get the Unemployment Rate and set them to be in the forexData dataframe.
unemploymentRate = interpolatedForFreq(
    quandl.get("SGE/EURUNR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31"))
forexData["UnemploymentRate"] = unemploymentRate
forexData = forexData.fillna(method='ffill')
forexData = forexData.fillna(0)
unemploymentRateScaler = MinMaxScaler(feature_range=(0, 1))
unemploy = np.array(forexData["UnemploymentRate"]).reshape(-1, 1)
unemploymentRateScaler.fit(consumerSpend)
forexData["UnemploymentRate"] = unemploymentRateScaler.transform(consumerSpend)
forexData = forexData.fillna(0)
print(forexData)

# Normalize the data so that it has the same noise of the model when we will had a few other variables into the mix.
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(forexData)
'''

forexData = forexData.fillna(0)
training_set_scaled = np.array(forexData)
print(training_set_scaled)

# Setting the training and testing length accordingly.
train_size = int(len(forexData) * 0.95)
test_size = len(forexData) - train_size
train_data = training_set_scaled[0:train_size]
test_data = training_set_scaled[:-train_size]

# Period length
period = 30

# Two arrays that will hold the referrence to the price and wheter it has increased in a predetermined time frame.
prices = []
toPredict = []

# Set a new array and make the array so that if a price is larger in 30 trading days it result in a 1 or a 0 is not.
for i in range(period, train_size + test_size):
    prices.append(training_set_scaled[i - period:i])
    toPredict.append(training_set_scaled[i, 0])



prices, toPredict = np.array(prices), np.array(toPredict)

prices = np.reshape(prices, (prices.shape[0], prices.shape[1], 8))

print(toPredict[0])
print(prices[0])
# Define the model.
model = Sequential()

# Add layers of node to the model.
model.add(LSTM(units=50, return_sequences=True, input_shape=(prices.shape[1], 8)))
model.add(LSTM(units=50))


model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(prices, toPredict, epochs=150, batch_size=100, validation_split=0.05)

# Graph the loss of both the training and the testing loss for analysis purposes.
print(history.history['loss'])
print(history.history['val_loss'])
plt.plot(history.history['loss'], color='blue')
plt.plot(history.history['val_loss'], color='green')
plt.title('Model train & Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Separate the code of the array that will constitute the test data, reshape and normalize it.
inputs = np.array(forexData[train_size:])
# inputs = inputs.reshape(-1, 1)
# inputs = scaler.transform(inputs)

# Empty array that will be filled with the past period of data used for the prediction which will be used for the prediction.
X_test = []
pricesToPredict = []
for i in range(period, test_size):
    X_test.append(inputs[i - period:i])
    pricesToPredict.append(inputs[i, 0])

print("\nX_Test: ", X_test)
print("\npricesToPredict: ", pricesToPredict)

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 8))
# Predict the prices using the model and do the inverse of the normalization done.
predicted_stock_price = model.predict(X_test)
predictedNormalPrice = predicted_stock_price  # forexScaler.inverse_transform(predicted_stock_price)

print("\n\npredictedNormalPrice: " , predictedNormalPrice)
# Calculate the buffer of the second array that need to be used for the real stock prices.
buffer = train_size + period

forexPrices = quandl.get("SGE/BELCUR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1993-12-31", end_date="2018-03-29")

# Reconvert the prices to display for the graph.
pricesTest = np.array(forexPrices[buffer:])
realPrices = np.array(forexPrices)



pricesTest = np.nan_to_num(pricesTest)
predictedNormalPrice = np.nan_to_num(predictedNormalPrice)

# Import the packages to make the linear regression that will be used for analysis purposes.
from sklearn.linear_model import LinearRegression

linearRegressionModel = LinearRegression()
linearRegressionModel.fit(prices[train_size:, 0], toPredict[train_size:])
linearReg = linearRegressionModel.predict(X_test[:, 0])
linearReg = np.array(linearReg).reshape(-1, 1)
linearReg = forexScaler.inverse_transform(linearReg)

from statsmodels.tsa.arima_model import ARIMA

# Method that gets the Arima prediction to Be Used for Analysis.
def getArimaPrediction(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


arimaPred = []
Actual = [x for x in training_set_scaled[0:train_size + period, 0]]

Actual = np.array(realPrices[:buffer])

print(Actual)

# Percentage variable used to show the progress of the following loop.
percentage = round(len(test_data-period)/100)
# Variable only used to count the loops, which is useful for progression display.
loopCount = 0
# For loop that makes the ARIMA prediction.
for i in range(buffer, len(forexPrices)):
    # Increment for proper display.
    loopCount = loopCount + 1
    # To display to know when the program will approximately finish.
    if(loopCount % percentage == 0 ):
        print("\nPercentage Done of Arima Precition = %f %% \n" % int((round(loopCount / percentage))))
    ActualValue = realPrices[i]
    # forcast value
    Prediction = getArimaPrediction(Actual, 3, 1, 0)
    print('Actual=%f, Predicted=%f, Difference=%f' % (ActualValue, Prediction, abs(ActualValue-Prediction)))
    # add it in the list
    arimaPred.append(Prediction)
    Actual = np.append(Actual, ActualValue)




print(arimaPred)
# Moving Average method to Be Used for Analysis.
def moving_average(y_true, period):

    y_true = np.ndarray.flatten(y_true)
    movingaveragepreds = []
    for i in range(period, len(y_true)):
        movingaveragepreds.append(np.mean(y_true[i - period:i]))

    return movingaveragepreds


# Reconvert the prices to display for the graph.
pricesForMoving = np.array(forexPrices[train_size:])
movingAverage = moving_average(pricesForMoving, period)

print(len(pricesTest), len(predictedNormalPrice), len(linearReg), len(movingAverage))

# Setup the display of the graphing.
plt.plot(pricesTest, color='black')
plt.plot(predictedNormalPrice, color='red')
plt.plot(arimaPred, color='yellow')
plt.plot(movingAverage, color='blue')
plt.title('Deep Learning Prediction of EURO/USD Exchange Rate')
plt.xlabel('Trading Days')
plt.ylabel('Foreign Exchange Rate EURO/USD')
plt.legend(['Real Forex', "LSTM", "ARIMA", "Moving Average"], loc='best')
plt.show()

plt.plot(abs(pricesTest - predictedNormalPrice))
plt.show()

# Setup the display of the graphing.
plt.plot(pricesTest, color='black')
plt.plot(predictedNormalPrice, color='red')

plt.title('Deep Learning Prediction of EURO/USD Exchange Rate')
plt.xlabel('Trading Days')
plt.ylabel('Foreign Exchange Rate EURO/USD')
plt.legend(['Real Forex', "LSTM"], loc='best')
plt.show()



# Will be used for the metrics calculations.
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# LSTM Metrics to be calculated and then printed.
lstmRMSE = round(math.sqrt(mean_squared_error(pricesTest, predictedNormalPrice)), 6)
lstmMAE = round(mean_absolute_error(pricesTest, predictedNormalPrice), 6)
lstmMAPE = round(mean_absolute_percentage_error(pricesTest, predictedNormalPrice), 6)
print("LSTM")
print("RMSE: ", lstmRMSE)
print("MAE: ", lstmMAE)
print("MAPE: ", lstmMAPE)

# Linear Regression Metrics to be calculated and then printed.
regRMSE = round(math.sqrt(mean_squared_error(pricesTest, arimaPred)), 6)
regMAE = round(mean_absolute_error(pricesTest, arimaPred), 6)
regMAPE = round(mean_absolute_percentage_error(pricesTest, arimaPred), 6)
print("Linear Regression")
print("RMSE: ", regRMSE)
print("MAE: ", regMAE)
print("MAPE: ", regMAPE)

# Linear Regression Metrics to be calculated and then printed.
movRMSE = round(math.sqrt(mean_squared_error(pricesTest, movingAverage)), 6)
movMAE = round(mean_absolute_error(pricesTest, movingAverage), 6)
movMAPE = round(mean_absolute_percentage_error(pricesTest, movingAverage), 6)
print("Moving Average")
print("RMSE: ", movRMSE)
print("MAE: ", movMAE)
print("MAPE: ", movMAPE)
