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
forexData = pd.DataFrame()

# Get the Forex data. Which is set to Euro to USD
forexPrices = quandl.get("CUR/GBP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31", end_date="2015-03-29")
forexScaler = MinMaxScaler(feature_range=(0, 1))
forexScaler.fit(forexPrices)
#forexData = pd.DataFrame(forexScaler.transform(forexPrices))
forexData = forexPrices
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

# Function used to interpolate the missing values for monthly values and others missing entries.
def interpolatedForFreqPercent(call):
    df = pd.DataFrame(call)
    upsampled = df.resample("B").ffill()
    df.replace([np.inf, -np.inf], np.nan)
    df = pd.DataFrame(upsampled)
    df = df.pct_change()
    df = df.fillna(method='bfill')
    df = df.fillna(0)
    return df

# Get the Interest Rate of GBR and set them to be in the forexData dataframe.
interestRateGBR = interpolatedForFreq(quandl.get("SGE/GBRIR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["InterestRate-GBR"] = interestRateGBR
forexData = forexData.fillna(method='ffill')
interestRateGBRScaler = MinMaxScaler(feature_range=(0, 1))
iRGBR = np.array(forexData["InterestRate-GBR"]).reshape(-1, 1)
interestRateGBRScaler.fit(iRGBR)
forexData["InterestRate-GBR"] = interestRateGBRScaler.transform(iRGBR)
'''
# Get the Inflation Rate of GBR and set them to be in the forexData dataframe.
inflationRateGBR = interpolatedForFreq(quandl.get("SGE/GBRCPIC", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["InflationRate-GBR"] = inflationRateGBR
forexData = forexData.fillna(method='ffill')
inflationRateGBRScaler = MinMaxScaler(feature_range=(0, 1))
iFGBR = np.array(forexData["InflationRate-GBR"]).reshape(-1, 1)
inflationRateGBRScaler.fit(iFGBR)
forexData["InflationRate-GBR"] = inflationRateGBRScaler.transform(iFGBR)
'''
# Get the Inflation Rate of AUSTRALIA and set them to be in the forexData dataframe.
inflationRateAUS = interpolatedForFreq(quandl.get("SGE/GBRCPIC", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["InflationRate-CAD"] = inflationRateAUS
forexData = forexData.fillna(method='ffill')
inflationRateAUSScaler = MinMaxScaler(feature_range=(0, 1))
iFAUS = np.array(forexData["InflationRate-CAD"]).reshape(-1, 1)
inflationRateAUSScaler.fit(iFAUS)
forexData["InflationRate-CAD"] = inflationRateAUSScaler.transform(iFAUS)

# Get the Imports of GBR and set them to be in the forexData dataframe.
importsGBR = interpolatedForFreqPercent(quandl.get("SGE/GBRIMVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData['Imports-GBR'] = importsGBR
forexData = forexData.fillna(method='ffill')
importsGBRScaler = MinMaxScaler(feature_range=(0, 1))
impGBR = np.array(forexData["Imports-GBR"]).reshape(-1, 1)
importsGBRScaler.fit(impGBR)
forexData["Imports-GBR"] = importsGBRScaler.transform(impGBR)

# Get the Exports of GBR and set them to be in the forexData dataframe.
exportsGBR = interpolatedForFreqPercent(quandl.get("SGE/GBREXVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["Exports-GBR"] = exportsGBR
forexData = forexData.fillna(method='ffill')
exportsGBRScaler = MinMaxScaler(feature_range=(0, 1))
expGBR = np.array(forexData["Exports-GBR"]).reshape(-1, 1)
exportsGBRScaler.fit(expGBR)
forexData["Exports-GBR"] = exportsGBRScaler.transform(expGBR)

# Get the gdp of GBR and set them to be in the forexData dataframe.
gdpGBR = interpolatedForFreqPercent(quandl.get("SGE/GBRG", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["GDP-GBR"] = gdpGBR
forexData = forexData.fillna(method='ffill')
gdpGBRScaler = MinMaxScaler(feature_range=(0, 1))
gdppGBR = np.array(forexData["GDP-GBR"]).reshape(-1, 1)
gdpGBRScaler.fit(gdppGBR)
forexData["GDP-GBR"] = gdpGBRScaler.transform(gdppGBR)

# Get the Consumer Spending of GBR and set them to be in the forexData dataframe.
consumerSpendingGBR = interpolatedForFreqPercent(quandl.get("SGE/GBRCSP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["ConsumerSpending-GBR"] = consumerSpendingGBR
forexData = forexData.fillna(method='ffill')
consumerSpendingGBRScaler = MinMaxScaler(feature_range=(0, 1))
consumerSpendGBR = np.array(forexData["ConsumerSpending-GBR"]).reshape(-1, 1)
consumerSpendingGBRScaler.fit(consumerSpendGBR)
forexData["ConsumerSpending-GBR"] = consumerSpendingGBRScaler.transform(consumerSpendGBR)

# Get the Unemployment Rate of GBR and set them to be in the forexData dataframe.
unemploymentRateGBR = interpolatedForFreq(quandl.get("SGE/GBRUNR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["UnemploymentRate-GBR"] = unemploymentRateGBR
forexData = forexData.fillna(method='ffill')
unemploymentRateGBRScaler = MinMaxScaler(feature_range=(0, 1))
unemployGBR = np.array(forexData["UnemploymentRate-GBR"]).reshape(-1, 1)
unemploymentRateGBRScaler.fit(unemployGBR)
forexData["UnemploymentRate-GBR"] = unemploymentRateGBRScaler.transform(unemployGBR)

# Get the CPI of the GBR and set them to be in the forexData dataframe.
consumerPriceIndexGBR = interpolatedForFreqPercent(quandl.get("SGE/GBRCPI", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["CPI-GBR"] = consumerPriceIndexGBR
forexData = forexData.fillna(method='ffill')
cpiGBRScaler = MinMaxScaler(feature_range=(0, 1))
cpiGBR = np.array(forexData["CPI-GBR"]).reshape(-1, 1)
cpiGBRScaler.fit(cpiGBR)
forexData["CPI-GBR"] = cpiGBRScaler.transform(cpiGBR)


# USA Data.

# Get the Interest Rate of the United States and set them to be in the forexData dataframe.
interestRateUSA = interpolatedForFreq(quandl.get("SGE/USAIR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["InterestRate-USA"] = interestRateUSA
forexData = forexData.fillna(method='ffill')
interestRateUSAScaler = MinMaxScaler(feature_range=(0, 1))
iRUSA = np.array(forexData["InterestRate-USA"]).reshape(-1, 1)
interestRateUSAScaler.fit(iRUSA)
forexData["InterestRate-USA"] = interestRateUSAScaler.transform(iRUSA)

# Get the Inflation Rate of the US and set them to be in the forexData dataframe.
inflationRateUSA = interpolatedForFreq(quandl.get("SGE/USACPIC", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["InflationRate-USA"] = inflationRateUSA
forexData = forexData.fillna(method='ffill')
inflationRateUSAScaler = MinMaxScaler(feature_range=(0, 1))
iF = np.array(forexData["InflationRate-USA"]).reshape(-1, 1)
inflationRateUSAScaler.fit(iF)
forexData["InflationRate-USA"] = inflationRateUSAScaler.transform(iF)




# Get the Imports of the US and set them to be in the forexData dataframe.
importsUSA = interpolatedForFreqPercent(quandl.get("SGE/USAIMVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData['Imports-USA'] = importsUSA
forexData = forexData.fillna(method='ffill')
importsUSAScaler = MinMaxScaler(feature_range=(0, 1))
impUS = np.array(forexData["Imports-USA"]).reshape(-1, 1)
importsUSAScaler.fit(impUS)
forexData["Imports-USA"] = importsUSAScaler.transform(impUS)

# Get the Exports of the US and set them to be in the forexData dataframe.
exportsUSA = interpolatedForFreqPercent(quandl.get("SGE/USAEXVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["Exports-USA"] = exportsUSA
forexData = forexData.fillna(method='ffill')
exportsUSAScaler = MinMaxScaler(feature_range=(0, 1))
expUS = np.array(forexData["Exports-USA"]).reshape(-1, 1)
exportsUSAScaler.fit(expUS)
forexData["Exports-USA"] = exportsUSAScaler.transform(expUS)

# Get the CPI of the US and set them to be in the forexData dataframe.
consumerPriceIndexUSA = interpolatedForFreqPercent(quandl.get("SGE/USACPI", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["CPI-USA"] = consumerPriceIndexUSA
forexData = forexData.fillna(method='ffill')
cpiUSAScaler = MinMaxScaler(feature_range=(0, 1))
cpiUS = np.array(forexData["CPI-USA"]).reshape(-1, 1)
cpiUSAScaler.fit(cpiUS)
forexData["CPI-USA"] = cpiUSAScaler.transform(cpiUS)

# Get the Unemployment Rate of the USA and set them to be in the forexData dataframe.
unemploymentRateUSA = interpolatedForFreq(quandl.get("SGE/USAUNR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["UnemploymentRate-USA"] = unemploymentRateUSA
forexData = forexData.fillna(method='ffill')
forexData = forexData.fillna(0)
unemploymentRateUSAScaler = MinMaxScaler(feature_range=(0, 1))
unemployUSA = np.array(forexData["UnemploymentRate-USA"]).reshape(-1, 1)
unemploymentRateUSAScaler.fit(unemployUSA)
forexData["UnemploymentRate-USA"] = unemploymentRateUSAScaler.transform(unemployUSA)

# Get the Consumer Spending of the USA and set them to be in the forexData dataframe.
consumerSpendingUSA = interpolatedForFreqPercent(quandl.get("SGE/USACSP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["ConsumerSpending-USA"] = consumerSpendingUSA
forexData = forexData.fillna(method='ffill')
consumerSpendingUSAScaler = MinMaxScaler(feature_range=(0, 1))
consumerSpendUSA = np.array(forexData["ConsumerSpending-USA"]).reshape(-1, 1)
consumerSpendingUSAScaler.fit(consumerSpendUSA)
forexData["ConsumerSpending-USA"] = consumerSpendingUSAScaler.transform(consumerSpendUSA)

# Get the gdp and set them to be in the forexData dataframe.
gdpUSA = interpolatedForFreqPercent(quandl.get("SGE/USAG", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31"))
forexData["GDP-USA"] = gdpUSA
forexData = forexData.fillna(method='ffill')
gdpUSAScaler = MinMaxScaler(feature_range=(0, 1))
gdppUSA = np.array(forexData["GDP-USA"]).reshape(-1, 1)
gdpUSAScaler.fit(gdppUSA)
forexData["GDP-USA"] = gdpUSAScaler.transform(gdppUSA)

forexPrices = quandl.get("CUR/GBP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31", end_date="2015-03-29")

forexData["RATE"] = forexScaler.transform(forexPrices)

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
#print(training_set_scaled)

# Setting the training and testing length accordingly.
train_size = int(len(forexData) * 0.90)
test_size = len(forexData) - train_size
train_data = training_set_scaled[0:train_size]
test_data = training_set_scaled[:-train_size]

# Period length
period = 30

# Two arrays that will hold the referrence to the price and wheter it has increased in a predetermined time frame.
prices = []
toPredict = []

# Set a new array and make the array so that if a price is larger in 30 trading days it result in a 1 or a 0 is not.
for i in range(period, train_size + test_size - 5):
    prices.append(training_set_scaled[i - period:i])
    toPredict.append(training_set_scaled[i + 5, 0])



prices, toPredict = np.array(prices), np.array(toPredict)

prices = np.reshape(prices, (prices.shape[0], prices.shape[1], 17))

# Define the model.
model = Sequential()

# Add layers of node to the model.
model.add(LSTM(units=38, return_sequences=True, input_shape=(prices.shape[1], 17)))
model.add(Dropout(0.05))
model.add(LSTM(units=38))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(prices, toPredict, epochs=150, batch_size=100, validation_split=0.1)

# Graph the loss of both the training and the testing loss for analysis purposes.

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
# inputs = forexscaler.transform(inputs)

# Empty array that will be filled with the past period of data used for the prediction which will be used for the prediction.
X_test = []
pricesToPredict = []
for i in range(period, test_size - 5):
    X_test.append(inputs[i - period:i])
    pricesToPredict.append(inputs[i + 5, 0])



X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 17))
# Predict the prices using the model and do the inverse of the normalization done.
predicted_stock_price = model.predict(X_test)
predictedNormalPrice = forexScaler.inverse_transform(predicted_stock_price)  # forexScaler.inverse_transform(predicted_stock_price)


# Calculate the buffer of the second array that need to be used for the real stock prices.
buffer = train_size + period

forexPrices = quandl.get("CUR/GBP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31", end_date="2015-03-29")

# Reconvert the prices to display for the graph.
pricesTest = np.array(forexPrices[buffer:])
realPrices = np.array(forexPrices)



pricesTest = np.nan_to_num(pricesTest)
predictedNormalPrice = np.nan_to_num(predictedNormalPrice)
lstmRMSE = round(math.sqrt(mean_squared_error(pricesTest[5:], predictedNormalPrice)), 6)
print("\nLSTM")
print("RMSE: ", lstmRMSE)



from statsmodels.tsa.arima_model import ARIMA

# Method that gets the Arima prediction to Be Used for Analysis.
def getArimaPrediction(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast(steps=5)
    return prediction[0][4]


arimaPred = []
Actual = [x for x in training_set_scaled[0:train_size + period, 0]]

Actual = np.array(realPrices[:buffer])


# Percentage variable used to show the progress of the following loop.
percentage = round(len(test_data-period)/100)
# Variable only used to count the loops, which is useful for progression display.
loopCount = 0
# For loop that makes the ARIMA prediction.
for i in range(buffer, len(forexPrices) - 5):
    # Increment for proper display.
    loopCount = loopCount + 1
    # To display to know when the program will approximately finish.
    if(loopCount % percentage == 0 ):
        print("\nPercentage Done of Arima Precition = %f %% \n" % int((round(loopCount / percentage))))
    ActualValue = realPrices[i]
    # forcast value
    Prediction = getArimaPrediction(Actual, 3, 2, 0)
    print('Actual=%f, Predicted=%f, Difference=%f' % (realPrices[i+5], Prediction, abs(ActualValue-Prediction)))
    # add it in the list
    arimaPred.append(Prediction)
    Actual = np.append(Actual, ActualValue)





# Moving Average method to Be Used for Analysis. NOT used in the final model.
def moving_average(y_true, period):

    y_true = np.ndarray.flatten(y_true)
    movingaveragepreds = []
    for i in range(period, len(y_true)):
        movingaveragepreds.append(np.mean(y_true[i - period:i]))

    return movingaveragepreds



# Reconvert the prices to display for the graph.
pricesForMoving = np.array(forexPrices[train_size:])
movingAverage = moving_average(pricesForMoving, period)

print("priceTest: ",pricesTest, "predictedNormalPrice" ,predictedNormalPrice, "movingAverage", movingAverage)

priceDataframe =  pd.DataFrame(pricesTest[:-5])
ema = priceDataframe.ewm(span=period , adjust=False).mean()
print(ema)

# Setup the display of the graphing.
plt.plot(pricesTest, color='black')
plt.plot(predictedNormalPrice, color='red')
plt.plot(arimaPred, color='grey')
plt.plot(movingAverage, color='blue')
plt.title('5 Day Forecast of the GBR/USD Exchange Rate')
plt.xlabel('Trading Days')
plt.ylabel('Foreign Exchange Value')
plt.legend(['Real Forex', "LSTM", "ARIMA", "Moving Average"], loc='best')
plt.show()


priceDataframe =  pd.DataFrame(pricesTest[:-5])
ema = priceDataframe.ewm(span=period , adjust=False).mean()
print(ema)

# Setup the display of the graphing.
plt.plot(pricesTest[-50:], color = 'black')
plt.plot(predictedNormalPrice[-50:], color = 'red')
plt.plot(arimaPred[-50:], color='gray', )
plt.plot(np.array(ema[-50:]), color= 'blue')
plt.show()


pricesTest = pricesTest[5:]

arimaErrors = abs(pricesTest - arimaPred)
emaErrors = abs(pricesTest - ema)
lstmErrors = abs(pricesTest - predictedNormalPrice)

plt.plot(arimaErrors, color = 'gray')
plt.plot(emaErrors, color='blue')
plt.plot(lstmErrors, color='red')
plt.legend([ "ARIMA Errors",'EMA Errors', "LSTM Errors"], loc='best')
plt.show()




# Will be used for the metrics calculations.
def mean_absolute_percentage_error(y_true, y_pred):
   # y_true, y_pred = check_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(predictedNormalPrice, pricesTest)

# LSTM Metrics to be calculated and then printed.
lstmRMSE = round(math.sqrt(mean_squared_error(pricesTest, predictedNormalPrice)), 6)
lstmMAE = round(mean_absolute_error(pricesTest, predictedNormalPrice), 6)
lstmMAPE = round(mean_absolute_percentage_error(pricesTest, predictedNormalPrice), 6)
print("LSTM")
print("RMSE: ", lstmRMSE)
print("MAE: ", lstmMAE)
print("MAPE: ", lstmMAPE)

# ARIMA  Metrics to be calculated and then printed.
regRMSE = round(math.sqrt(mean_squared_error(pricesTest, arimaPred)), 6)
regMAE = round(mean_absolute_error(pricesTest, arimaPred), 6)
regMAPE = round(mean_absolute_percentage_error(pricesTest, arimaPred), 6)
print("\nARIMA")
print("RMSE: ", regRMSE)
print("MAE: ", regMAE)
print("MAPE: ", regMAPE)

# Moving Average Metrics to be calculated and then printed.
movRMSE = round(math.sqrt(mean_squared_error(pricesTest, ema)), 6)
movMAE = round(mean_absolute_error(pricesTest, ema), 6)
movMAPE = round(mean_absolute_percentage_error(pricesTest, ema), 6)
print("\nMoving Average")
print("RMSE: ", movRMSE)
print("MAE: ", movMAE)
print("MAPE: ", movMAPE)
