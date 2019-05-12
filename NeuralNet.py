import pandas as pd
import numpy as np
import talib as ta
from talib.abstract import *
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

import os
from Alpha_vantage_pricing import get_daily
from Alpha_vantage_pricing import get_intraday


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)


tickers = ['FB'] #, 'BA','AMZN', 'AAPL', 'MSFT']# 'XOM', 'JNJ', 'GE'] # 'BRK-B', 'T', 'VZ']
stocks = {}

for tick in tickers:
    df = pd.DataFrame(get_daily(tick,'full'))
    df.index = pd.to_datetime(df.timestamp)
    df = df.drop('timestamp', axis=1)
    stocks[tick] = df[::-1]

for j in stocks:
    plt.figure(figsize = (18,9))
    plt.title(str(j))
    plt.plot(range(stocks[j].shape[0]),(stocks[j].low+stocks[j].high)/2.0)
    plt.xticks(range(0,stocks[j].shape[0],500),stocks[j].index[::500],rotation=30, fontsize = 6)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.show()

#high_prices = df.loc[:,'high'].values
#low_prices = df.loc[:,'low'].values
#mid_prices = (high_prices+low_prices)/2.0

for i,j in enumerate(stocks):
    stocks[j].columns = [s.lower() for s in stocks[j].columns]
    stocks[j].volume = stocks[j].volume.apply(lambda x: float(x))

def get_indicators(stocks, period):
    stocks_indicators = {}
    for i in stocks:
        features = pd.DataFrame(ta.SMA(stocks[i].close.values, timeperiod=5))
        features.columns = ['sma_5']
        features['sma_10'] = pd.DataFrame(ta.SMA(stocks[i].close.values, timeperiod=10))
        features['mom_10'] = pd.DataFrame(ta.MOM(stocks[i].close.values,10))
        features['wma_10'] = pd.DataFrame(ta.WMA(stocks[i].close.values,10))
        features['wma_5'] = pd.DataFrame(ta.WMA(stocks[i].close.values,5))
        features['fastk'], features['fastd'] = ta.STOCHF(stocks[i].high.values,stocks[i].low.values, stocks[i].close.values, fastk_period=14, fastd_period=3)
        features['macd'], features['macdSignal1'], features['macdHist1'] = ta.MACD(stocks[i].close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        features['rsi'] = pd.DataFrame(ta.RSI(stocks[i].close.values, timeperiod=14))
        features['willr'] = pd.DataFrame(ta.WILLR(stocks[i].high.values, stocks[i].low.values, stocks[i].close.values, timeperiod=14))
        features['cci'] = pd.DataFrame(ta.CCI(stocks[i].high.values, stocks[i].low.values, stocks[i].close.values, timeperiod=14))
        features['adosc'] = pd.DataFrame(ta.ADOSC(stocks[i].high.values, stocks[i].low.values, stocks[i].close.values, stocks[i].volume.values, fastperiod=3, slowperiod=10))
        features['close'] = pd.DataFrame(stocks[i].close.values)
        features['pct_change'] = ta.ROC(stocks[i].close.values, timeperiod=period)
        features['pct_change'] = features['pct_change'].shift(-period)
        features['pct_change'] = features['pct_change'].apply(lambda x: '1' if x > 0 else '0' if x <= 0 else np.nan)
        features.index = stocks[i].index
        features = features.dropna()
        #features = features.iloc[np.where(features.index=='1998-5-5')[0][0]:np.where(features.index=='2015-5-5')[0][0]]
        stocks_indicators[i] = features
    return stocks_indicators


#create model


stocks_indicators = get_indicators(stocks,5)
for j in stocks:
    X = stocks_indicators[j].iloc[:, :-1].astype('float')
    y = stocks_indicators[j].iloc[:, -1].astype('float')
    tscv = TimeSeriesSplit(n_splits=2)
    tscv.get_n_splits(X)
    for train_index, test_index in tscv.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        classifier = Sequential()
        classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu', input_dim=X.shape[1]))
        classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
        y_pred = classifier.predict(X_test)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        print(str(j))
        print(classification_report(y_test,pd.DataFrame(y_pred)))
        print(confusion_matrix(y_test,pd.DataFrame(y_pred)))
        print(accuracy_score(y_test, y_pred))


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
for j in stocks:
    stockdata = stocks_indicators[j].astype('float')
    tscv = TimeSeriesSplit(n_splits=2)
    tscv.get_n_splits(stockdata)
    for train_index, test_index in tscv.split(stockdata):
        # print("TRAIN:", train_index, "TEST:", test_index)
        training_set, testing_set = stockdata.iloc[train_index], stockdata.iloc[test_index]

    training_set_scaled = sc.fit_transform(training_set)
    # Creating a data structure with 60 timesteps, 14 features, and 1 output (i.e. N samples of 60 timesteps of all features) like samples of 60 length time series
    X_train = []
    y_train = []
    for i in range(20, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-20:i, :15])
        y_train.append(training_set_scaled[i, 15])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping for 3D
    #X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Initialising the RNN
    classifier = Sequential()
    classifier.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1:3])))
    classifier.add(Dropout(0.5))
    classifier.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1:3])))
    classifier.add(Dropout(0.5))
    classifier.add(Flatten())
    classifier.add(Dense(20, activation = 'relu'))
    # Adding a second LSTM layer and some Dropout regularisation
    #classifier.add(LSTM(units=128, return_sequences=True))
    #classifier.add(Dropout(0.2))
    #classifier.add(LSTM(units=64, return_sequences=True))
    #classifier.add(Dropout(0.2))
    #classifier.add(LSTM(units=32))
    #classifier.add(Dropout(0.2))

    classifier.add(Dense(units=1, activation='sigmoid'))
    # Compiling the RNN
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    #classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Fitting the RNN to the Training set
    classifier.fit(X_train, y_train, epochs=250, batch_size=10)



testing_set_scaled = sc.transform(testing_set)
len(testing_set)
X_test = []
y_test = []
for i in range(20, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-20:i, :15])
    y_test.append(testing_set_scaled[i, 15])

X_test, y_test = np.array(X_test), np.array(y_test)

# Reshaping for 3D
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

y_pred = classifier.predict(X_test)
#reverse the scaling
#y_pred = sc.inverse_transform(predicted_stock_price)

t=[]
for i in range(0, len(y_test)):
    t.append(y_pred.item(i))

test = pd.DataFrame()
test['pred']=pd.Series(t)
test['test'] = y_test
test['pred0'] = 0
test.pred0[test['pred'] > .5] = 1
print(classification_report(test.test,test.pred0))
print(confusion_matrix(test.test,test.pred0))

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,test.pred0)
np.sqrt(mean_squared_error(y_test,test.pred0))
