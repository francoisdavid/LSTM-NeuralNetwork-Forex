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



# Get the Forex data. Which is set to Euro to USD
AUS  = np.array(quandl.get("SGE/AUSCUR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31", end_date="2018-03-29"))
CAD  = np.array(quandl.get("SGE/CANCUR", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1992-12-31", end_date="2018-03-29"))
GBR = np.array(quandl.get("CUR/GBP", authtoken="g5x4nVyzgx-hKs6s7Nt2", start_date="1999-12-31", end_date="2015-03-29"))


CAD = CAD.reshape(-1)
AUS = AUS.reshape(-1)
GBR = GBR.reshape(-1)


print("Length: ")
cadL = int(len(CAD))
gbrL = int(len(GBR))
ausL = int(len(AUS))


print("CAD: " , len(CAD), len(CAD)*.9, len(CAD)*.1)
print("BGR: ", len(GBR), len(GBR)*.9, len(GBR)*.1)
print("AUS: ", len(AUS), len(AUS)*.9, len(AUS)*.1)


trainCAD = CAD[:int(cadL * 0.9)]
testCAD = CAD[int(cadL*.9):]
trainGBR = GBR[:int(gbrL * 0.9)]
testGBR = GBR[int(gbrL*.9):]
trainAUS = AUS[:int(ausL * 0.9)]
testAUS = AUS[int(ausL*.9):]

print("\n\nMeanCAD ",statistics.mean(CAD),"TrainMeanCAD: ", statistics.mean(trainCAD), "TestMeanCAD: ", statistics.mean(testCAD))
print("MeanGBR",statistics.mean(GBR),"TrainMeanCAD: ", statistics.mean(trainGBR), "TestMeanCAD: ", statistics.mean(testGBR))
print("MeanAUS",statistics.mean(AUS),"TrainMeanCAD: ", statistics.mean(trainAUS), "TestMeanCAD: ", statistics.mean(testAUS))


print("\n\nMedianCAD ", statistics.median(CAD),"TrainMedCAD: ", statistics.median(trainCAD), "TestMedCAD: ", statistics.median(testCAD))
print("MedianGBR", statistics.median(GBR),"TrainMedCAD: ", statistics.median(trainGBR), "TestMedCAD: ", statistics.median(testGBR))
print("MedianAUS", statistics.median(AUS),"TrainMedCAD: ", statistics.median(trainAUS), "TestMedCAD: ", statistics.median(testAUS))

print("\n\nMaxCAD ", max(CAD),"TrainMaxCAD: ", max(trainCAD), "TestMaxCAD: ", max(testCAD))
print("MaxGBR", max(GBR),"TrainMaxCAD: ", max(trainGBR), "TestMaxCAD: ", max(testGBR))
print("MaxAUS", max(AUS),"TrainMaxCAD: ", max(trainAUS), "TestMaxCAD: ", max(testAUS))


print("\n\nMinCAD ", min(CAD),"TrainMinCAD: ", min(trainCAD), "TestMinCAD: ", min(testCAD))
print("MinGBR", min(GBR),"TrainMinCAD: ", min(trainGBR), "TestMinCAD: ", min(testGBR))
print("MinAUS", min(AUS),"TrainMinCAD: ", min(trainAUS), "TestMinCAD: ", min(testAUS))

