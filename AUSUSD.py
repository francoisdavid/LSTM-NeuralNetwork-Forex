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


# DataFrame that will be filled with the information acquired from Quandl.
forexData = pd.DataFrame()

# Get the Forex data. Which is set to Euro to USD
forexPrices  = quandl.get("SGE/AUSCUR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31", end_date="2018-03-29")
forexScaler = MinMaxScaler(feature_range=(0, 1))
forexScaler.fit(forexPrices)
forexData = forexPrices
#forexData = pd.DataFrame(forexScaler.transform(forexPrices))
dataLength = len(forexPrices)

# Function used to interpolate the missing values for monthly values and others missing entries.
def interpolatedForFreq(call):
    df = pd.DataFrame(call)
    upsampled = df.resample("B").ffill()
    df = pd.DataFrame(upsampled)
    df = df.pct_change()
    df = df.fillna(method='bfill')
    df = df.fillna(0)
    return df

# Function used to interpolate the missing values for monthly values and others missing entries.
def interpolatedForFreqPercent(call):
    df = pd.DataFrame(call)
    upsampled = df.resample("B").ffill()
    df = pd.DataFrame(upsampled)
    df = df.fillna(method='bfill')
    df = df.fillna(0)
    return df


# AUSTRALIA Data

#Get the Interest Rate of AUSTRALIA and set them to be in the forexData dataframe.
interestRateAUS = interpolatedForFreq(quandl.get("SGE/AUSIR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["InterestRate-AUS"] = interestRateAUS
forexData = forexData.fillna(method='ffill')
interestRateAUSScaler = MinMaxScaler(feature_range=(0, 1))
iRAUS = np.array(forexData["InterestRate-AUS"]).reshape(-1, 1)
interestRateAUSScaler.fit(iRAUS)
forexData["InterestRate-AUS"] = interestRateAUSScaler.transform(iRAUS)

# Get the Inflation Rate of AUSTRALIA and set them to be in the forexData dataframe.
inflationRateAUS = interpolatedForFreq(quandl.get("SGE/AUSCPIC", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["InflationRate-AUS"] = inflationRateAUS
forexData = forexData.fillna(method='ffill')
inflationRateAUSScaler = MinMaxScaler(feature_range=(0, 1))
iFAUS = np.array(forexData["InflationRate-AUS"]).reshape(-1, 1)
inflationRateAUSScaler.fit(iFAUS)
forexData["InflationRate-AUS"] = inflationRateAUSScaler.transform(iFAUS)

# Get the Imports of AUSTRALIA and set them to be in the forexData dataframe.
importsAUS = interpolatedForFreq(quandl.get("SGE/AUSIMVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData['Imports-AUS'] = importsAUS
forexData = forexData.fillna(method='ffill')
importsAUSScaler = MinMaxScaler(feature_range=(0, 1))
impAUS = np.array(forexData["Imports-AUS"]).reshape(-1, 1)
importsAUSScaler.fit(impAUS)
forexData["Imports-AUS"] = importsAUSScaler.transform(impAUS)

# Get the Exports of AUSTRALIA and set them to be in the forexData dataframe.
exportsAUS = interpolatedForFreq(quandl.get("SGE/AUSEXVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["Exports-AUS"] = exportsAUS
forexData = forexData.fillna(method='ffill')
exportsAUSScaler = MinMaxScaler(feature_range=(0, 1))
expAUS = np.array(forexData["Exports-AUS"]).reshape(-1, 1)
exportsAUSScaler.fit(expAUS)
forexData["Exports-AUS"] = exportsAUSScaler.transform(expAUS)

# Get the gdp of AUSTRALIA and set them to be in the forexData dataframe.
gdpAUS = interpolatedForFreq(quandl.get("SGE/AUSG", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["GDP-AUS"] = gdpAUS
forexData = forexData.fillna(method='ffill')
gdpAUSScaler = MinMaxScaler(feature_range=(0, 1))
gdppAUS = np.array(forexData["GDP-AUS"]).reshape(-1, 1)
gdpAUSScaler.fit(gdppAUS)
forexData["GDP-AUS"] = gdpAUSScaler.transform(gdppAUS)

# Get the Consumer Spending of AUSTRALIA and set them to be in the forexData dataframe.
consumerSpendingAUS = interpolatedForFreq(quandl.get("SGE/AUSCSP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["ConsumerSpending-AUS"] = consumerSpendingAUS
forexData = forexData.fillna(method='ffill')
consumerSpendingAUSScaler = MinMaxScaler(feature_range=(0, 1))
consumerSpendAUS = np.array(forexData["ConsumerSpending-AUS"]).reshape(-1, 1)
consumerSpendingAUSScaler.fit(consumerSpendAUS)
forexData["ConsumerSpending-AUS"] = consumerSpendingAUSScaler.transform(consumerSpendAUS)

# Get the Unemployment Rate of AUSTRALIA and set them to be in the forexData dataframe.
unemploymentRateAUS = interpolatedForFreq(quandl.get("SGE/AUSUNR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["UnemploymentRate-AUS"] = unemploymentRateAUS
forexData = forexData.fillna(method='ffill')
unemploymentRateAUSScaler = MinMaxScaler(feature_range=(0, 1))
unemployAUS = np.array(forexData["UnemploymentRate-AUS"]).reshape(-1, 1)
unemploymentRateAUSScaler.fit(unemployAUS)
forexData["UnemploymentRate-AUS"] = unemploymentRateAUSScaler.transform(unemployAUS)

# Get the CPI of the AUSTRALIA and set them to be in the forexData dataframe.
consumerPriceIndexAUS = interpolatedForFreq(quandl.get("SGE/AUSCPI", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["CPI-AUS"] = consumerPriceIndexAUS
forexData = forexData.fillna(method='ffill')
cpiAUSScaler = MinMaxScaler(feature_range=(0, 1))
cpiAUS = np.array(forexData["CPI-AUS"]).reshape(-1, 1)
cpiAUSScaler.fit(cpiAUS)
forexData["CPI-AUS"] = cpiAUSScaler.transform(cpiAUS)



# USA DATA

# Get the Interest Rate of the United States and set them to be in the forexData dataframe.
interestRateUSA = interpolatedForFreq(quandl.get("SGE/USAIR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["InterestRate-USA"] = interestRateUSA
forexData = forexData.fillna(method='ffill')
interestRateUSAScaler = MinMaxScaler(feature_range=(0, 1))
iRUSA = np.array(forexData["InterestRate-USA"]).reshape(-1, 1)
interestRateUSAScaler.fit(iRUSA)
forexData["InterestRate-USA"] = interestRateUSAScaler.transform(iRUSA)

# Get the Inflation Rate of the US and set them to be in the forexData dataframe.
inflationRateUSA = interpolatedForFreqPercent(quandl.get("SGE/USACPIC", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["InflationRate-USA"] = inflationRateUSA
forexData = forexData.fillna(method='ffill')
inflationRateUSAScaler = MinMaxScaler(feature_range=(0, 1))
iF = np.array(forexData["InflationRate-USA"]).reshape(-1, 1)
inflationRateUSAScaler.fit(iF)
forexData["InflationRate-USA"] = inflationRateUSAScaler.transform(iF)

# Get the Imports of the US and set them to be in the forexData dataframe.
importsUSA = interpolatedForFreq(quandl.get("SGE/USAIMVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData['Imports-USA'] = importsUSA
forexData = forexData.fillna(method='ffill')
importsUSAScaler = MinMaxScaler(feature_range=(0, 1))
impUS = np.array(forexData["Imports-USA"]).reshape(-1, 1)
importsUSAScaler.fit(impUS)
forexData["Imports-USA"] = importsUSAScaler.transform(impUS)

# Get the Exports of the US and set them to be in the forexData dataframe.
exportsUSA = interpolatedForFreq(quandl.get("SGE/USAEXVOL", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["Exports-USA"] = exportsUSA
forexData = forexData.fillna(method='ffill')
exportsUSAScaler = MinMaxScaler(feature_range=(0, 1))
expUS = np.array(forexData["Exports-USA"]).reshape(-1, 1)
exportsUSAScaler.fit(expUS)
forexData["Exports-USA"] = exportsUSAScaler.transform(expUS)

# Get the CPI of the US and set them to be in the forexData dataframe.
consumerPriceIndexUSA = interpolatedForFreq(quandl.get("SGE/USACPI", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["CPI-USA"] = consumerPriceIndexUSA
forexData = forexData.fillna(method='ffill')
cpiUSAScaler = MinMaxScaler(feature_range=(0, 1))
cpiUS = np.array(forexData["CPI-USA"]).reshape(-1, 1)
cpiUSAScaler.fit(cpiUS)
forexData["CPI-USA"] = cpiUSAScaler.transform(cpiUS)

# Get the Unemployment Rate of the USA and set them to be in the forexData dataframe.
unemploymentRateUSA = interpolatedForFreq(quandl.get("SGE/USAUNR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["UnemploymentRate-USA"] = unemploymentRateUSA
forexData = forexData.fillna(method='ffill')
forexData = forexData.fillna(0)
unemploymentRateUSAScaler = MinMaxScaler(feature_range=(0, 1))
unemployUSA = np.array(forexData["UnemploymentRate-USA"]).reshape(-1, 1)
unemploymentRateUSAScaler.fit(unemployUSA)
forexData["UnemploymentRate-USA"] = unemploymentRateUSAScaler.transform(unemployUSA)

# Get the Consumer Spending of the USA and set them to be in the forexData dataframe.
consumerSpendingUSA = interpolatedForFreq(quandl.get("SGE/USACSP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["ConsumerSpending-USA"] = consumerSpendingUSA
forexData = forexData.fillna(method='ffill')
consumerSpendingUSAScaler = MinMaxScaler(feature_range=(0, 1))
consumerSpendUSA = np.array(forexData["ConsumerSpending-USA"]).reshape(-1, 1)
consumerSpendingUSAScaler.fit(consumerSpendUSA)
forexData["ConsumerSpending-USA"] = consumerSpendingUSAScaler.transform(consumerSpendUSA)

# Get the gdp and set them to be in the forexData dataframe.
gdpUSA = interpolatedForFreq(quandl.get("SGE/USAG", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31"))
forexData["GDP-USA"] = gdpUSA
forexData = forexData.fillna(method='ffill')
gdpUSAScaler = MinMaxScaler(feature_range=(0, 1))
gdppUSA = np.array(forexData["GDP-USA"]).reshape(-1, 1)
gdpUSAScaler.fit(gdppUSA)
forexData["GDP-USA"] = gdpUSAScaler.transform(gdppUSA)


# Get the Forex data. Which is set to Euro to USD
forexPrices  = quandl.get("SGE/AUSCUR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31", end_date="2018-03-29")

# Transform the value of the exchange rate so that it is normalized according to the range.
# It is done here to keep the index of the dataframe earlier and match the appropriate data with the date. 
forexData['Value'] =forexScaler.transform(forexPrices)


# Make sure none of the data entry are empty.
forexData = forexData.fillna(0)
print(forexData)
print("Is any value nan:", forexData.isnull().values.any())


# Normalize the data so that it has the same noise of the model when we will had a few other variables into the mix.
'''from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(forexData)'''
training_set_scaled = np.array(forexData)
print(training_set_scaled[0])

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

print(toPredict)
prices, toPredict = np.array(prices), np.array(toPredict)

prices = np.reshape(prices, (prices.shape[0], prices.shape[1], 17))


# Define the model.
model = Sequential()

# Add layers of node to the model.
model.add(LSTM(units=60, return_sequences=True, input_shape=(prices.shape[1], 17)))
model.add(Dropout(0.05))
model.add(LSTM(units=60))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

history = model.fit(prices, toPredict, epochs =150, batch_size = 100, validation_split=0.05)

# Graph the loss of both the training and the testing loss for analysis purposes.
print(history.history['loss'])
print(history.history['val_loss'])
plt.plot(history.history['loss'], color='blue')
plt.plot(history.history['val_loss'],   color = 'green')
plt.title('Model train & Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()




# Separate the code of the array that will constitute the test data, reshape and normalize it.
inputs = np.array(forexData[train_size:])


# Empty array that will be filled with the past period of data used for the prediction which will be used for the prediction.
X_test = []
pricesToPredict = []
for i in range(period, test_size):
    X_test.append(inputs[i - period:i])
    pricesToPredict.append(inputs[i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 17))
# Predict the prices using the model and do the inverse of the normalization done.
predicted_stock_price = model.predict(X_test)
print(predicted_stock_price)
predictedNormalPrice = forexScaler.inverse_transform(predicted_stock_price)


# Calculate the buffer of the second array that need to be used for the real stock prices.
buffer = train_size + period

forexPrices  = quandl.get("SGE/AUSCUR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31", end_date="2018-03-29")

# Reconvert the prices to display for the graph.
pricesTest = np.array(forexPrices[buffer:])
realPrices = np.array(forexPrices)
lstmRMSE = round(math.sqrt(mean_squared_error(pricesTest, predictedNormalPrice)), 6)
print("RMSE: ", lstmRMSE)
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
    Prediction = getArimaPrediction(Actual, 3, 2, 0)
    print('Actual=%f, Predicted=%f, Difference=%f' % (ActualValue, Prediction, abs(ActualValue-Prediction)))
    # add it in the list
    arimaPred.append(Prediction)
    Actual = np.append(Actual, ActualValue)


def moving_average(y_true, period):
    y_true = np.ndarray.flatten(y_true)
    movingAveragePreds = []
    for i in range(period, len(y_true)):
        movingAveragePreds.append(statistics.mean(y_true[i-period:i]) )
    return movingAveragePreds


# Reconvert the prices to display for the graph.
pricesForMoving = np.array(forexPrices[train_size:])
movingAverage = moving_average(pricesForMoving, period)

print("priceTest: ",pricesTest, "predictedNormalPrice" ,predictedNormalPrice, "movingAverage", movingAverage)

# Setup the display of the graphing.
plt.plot(pricesTest, color = 'black')
plt.plot(predictedNormalPrice, color = 'red')
plt.plot(arimaPred, color='gray', )
plt.plot(movingAverage, color= 'blue')
plt.title('Deep Learning Prediction of AUS/USD Exchange Rate')
plt.xlabel('Trading Days')
plt.ylabel('Foreign Exchange Rate AUS/USD')
plt.legend(['Real Forex', "LSTM", "Arima", "Moving Average"], loc='best')
plt.show()

plt.plot(abs(pricesTest - predictedNormalPrice))
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
movRMSE = round(math.sqrt(mean_squared_error(pricesTest, movingAverage)), 6)
movMAE = round(mean_absolute_error(pricesTest, movingAverage), 6)
movMAPE = round(mean_absolute_percentage_error(pricesTest, movingAverage), 6)
print("\nMoving Average")
print("RMSE: ", movRMSE)
print("MAE: ", movMAE)
print("MAPE: ", movMAPE)







