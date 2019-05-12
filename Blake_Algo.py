import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import os
from Alpha_vantage_pricing import get_daily
from Alpha_vantage_pricing import get_intraday

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

tickers = ['AAPL', 'UAA', 'EA', 'AMD', 'BLK', 'BA', 'CBRE', 'KO', 'GE', 'ECL', 'GS', 'KR', 'MRK']
for tick in tickers:
    df = pd.DataFrame(get_daily(tick,'full'))
    df.index = pd.to_datetime(df.timestamp)
    df = df.drop('timestamp', axis =1)
    df = df[::-1]

    df['sma_5']= pd.Series(df['close']).rolling(5).mean()
    df['sma_10']= pd.Series(df['close']).rolling(10).mean()
    df['sma_20']= pd.Series(df['close']).rolling(20).mean()
    df['MOM_5'] = ta.MOM(df.close.values, timeperiod=5)
    df['MOM_10'] = ta.MOM(df.close.values, timeperiod=10)
    df['EMA5'] = ta.EMA(df.close.values, timeperiod=5)
    df['EMA10'] = ta.EMA(df.close.values, timeperiod=10)
    df['EMA20'] = ta.EMA(df.close.values, timeperiod=20)
    df['EMA60'] = ta.EMA(df.close.values, timeperiod=60)
    #start adding indicators
    df['ROC'] = ta.ROC(df['close'].values, timeperiod=10)
    df['ROCP'] = ta.ROCP(df['close'].values, timeperiod=10)

    df['macd1'], df['macdSignal1'], df['macdHist1'] = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['SAR']= ta.SAR((df['high'].values), (df['low'].values), acceleration = 0.02, maximum = 0.2)
    df['slowk'], df['slowd'] = ta.STOCH((df['high'].values), (df['low'].values), (df['close'].values)
                                    ,fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['fastk'], df['fastd'] = ta.STOCHF((df['high'].values), (df['low'].values), (df['close'].values)
                                     , fastk_period=5, fastd_period=3, fastd_matype=0)
    df['RSI'] = ta.RSI(df.close.values, timeperiod=14)
    df['Williams'] = ta.WILLR(df.high.values, df.low.values, df.close.values, timeperiod=14)
    df['MFI'] = ta.MFI(df.high.values, df.low.values, df.close.values, df.volume.values.astype(float), timeperiod=12)
    df['NAT'] = ta.NATR(df.high.values, df.low.values, df.close.values, timeperiod=12)
    df['TRIX'] = ta.TRIX(df.close.values, timeperiod=30)
    df['PLUS_DI'] = ta.PLUS_DI(df.high.values, df.low.values, df.close.values, timeperiod=12)
    df['MINUS_DI'] = ta.MINUS_DI(df.high.values, df.low.values, df.close.values, timeperiod=12)
    df['BOP'] = ta.BOP(df.open.values, df.high.values, df.low.values, df.close.values)
    df['ADX'] = ta.ADX(df.high.values, df.low.values, df.close.values, timeperiod=12)
    df['TRANGE']= ta.TRANGE(df.high.values, df.low.values, df.close.values)
    df['OBV'] = ta.OBV(df.close.values, df.volume.values.astype(float))
    df['ADOSC'] = ta.ADOSC(df.high.values, df.low.values, df.close.values, df.volume.values.astype(float), fastperiod=3, slowperiod=10)
    df['CRTDR'] = (df.close.values - df.low.values) / (df.high.values - df.low.values)
    df['ATR'] = ta.ATR(df.high.values, df.low.values, df.close.values, timeperiod=12)

    #look at returns compared to x days back (pos shift to go back x days)
    df['1day_returns'] = np.log(df.close/df.close.shift(1))
    df['5day_returns'] = np.log(df.close/df.close.shift(5))
    df['10day_returns'] = np.log(df.close/df.close.shift(10))
    df['20day_returns'] = np.log(df.close/df.close.shift(20))
    df['60day_returns'] = np.log(df.close/df.close.shift(60))

    #Signals if price goes up or down in future (negative shift to go to future)
    df['1day_Signal'] = 0
    df['5day_Signal'] = 0
    df['10day_Signal'] = 0
    df['20day_Signal'] = 0
    df['60day_Signal'] = 0

    df['1day_Signal'][df['1day_returns'].shift(-1) > 0] = 1
    df['1day_Signal'][df['1day_returns'].shift(-1) < 0] = -1
    df['5day_Signal'][df['5day_returns'].shift(-5) > 0] = 1
    df['5day_Signal'][df['5day_returns'].shift(-5) < 0] = -1
    df['10day_Signal'][df['10day_returns'].shift(-10) > 0] = 1
    df['10day_Signal'][df['10day_returns'].shift(-10) < 0] = -1
    df['20day_Signal'][df['20day_returns'].shift(-20) > 0] = 1
    df['20day_Signal'][df['20day_returns'].shift(-20) < 0] = -1
    df['60day_Signal'][df['60day_returns'].shift(-60) > 0] = 1
    df['60day_Signal'][df['60day_returns'].shift(-60) < 0] = -1


    #plot market returns - only for the plot not
    df['mkt_returns1'] = df['1day_returns']    #.expanding().sum()
    df['mkt_returns5'] = df['5day_returns']   #.expanding().sum()
    df['mkt_returns10'] = df['10day_returns']   #.expanding().sum()
    df['mkt_returns20'] = df['20day_returns'] #.expanding().sum()
    df['mkt_returns60'] = df['60day_returns']  #.expanding().sum()

    #Confirm by looking at " perfect strategy returns" i.e. buy if price will go up in future
    #for today, get signal from 60 days ago and mult by todays 60 day ret
    df['str_returns1'] = df['1day_Signal'].shift(1)* df['1day_returns']
    df['str_returns5'] = df['5day_Signal'].shift(5)* df['5day_returns']
    df['str_returns10'] = df['10day_Signal'].shift(10)* df['10day_returns']
    df['str_returns20'] = df['20day_Signal'].shift(20)* df['20day_returns']
    df['str_returns60'] = df['60day_Signal'].shift(60)* df['60day_returns']
    Periods = ['1','5','10','20','60']
    for PER in Periods:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(df['str_returns'+str(PER)].iloc[::int(PER)].expanding().sum(), color='r', label=str(PER)+" day Perfect Strategy Returns")
        ax.plot(df['mkt_returns'+str(PER)].iloc[::int(PER)].expanding().sum(), color='b', label=str(PER)+" day Market Returns")
        plt.legend(loc='best')
        plt.title("Returns for Perfect Trading at the Close")
        plt.show()
    df.replace(-0,0)
    data = df.astype(float)

####NEED TO CONTINUE TO SPLIT OUT DATA INTO DIFF TIME PERIODS _ PREV WAS PREDICTING 60 day return one day out i.e today day 59 and predict day 60 returns
    SIGNAL = ['60', '20', '10', '5', '1']
    for Sig in SIGNAL:
        data = df.astype(float) #these vars are good bc we based the signal decision on the next day returns i.e. shift forward is implemented
        data = data[[str(Sig)+'day_Signal','open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'sma_20', 'MOM_5', 'MOM_10', 'EMA5', 'EMA10', 'EMA20', 'EMA60', 'macd1', 'macdSignal1', 'macdHist1', 'SAR', 'slowk', 'slowd', 'fastk', 'fastd', 'RSI', 'Williams', 'MFI', 'NAT', 'TRIX', 'PLUS_DI', 'MINUS_DI', 'BOP', 'ADX', 'TRANGE', 'OBV', 'ADOSC', 'CRTDR', 'ATR', '1day_returns', '5day_returns', '10day_returns', '20day_returns', '60day_returns']]
        data[str(Sig)+'day_Signal'][data[str(Sig)+'day_Signal'] == 1] = 'BUY'
        data[str(Sig)+'day_Signal'][data[str(Sig)+'day_Signal'] == -1] = 'SELL'
        data[str(Sig)+'day_Signal'][data[str(Sig)+'day_Signal'] == 0] = 'NA'
        data = data.dropna()
        X, y = data.iloc[:, 1:], data[str(Sig)+'day_Signal']
        X_train, X_test = X[X.index < '2017-01-01'], X[X.index >= '2017-01-01']
        y_train, y_test = y[y.index < '2017-01-01'], y[y.index >= '2017-01-01']
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        #y_test = np.log(y_test) gives shitty results
        #y_train = np.log(y_train)
        '''scalery = StandardScaler().fit(np.array(y_train).reshape(-1, 1))
        y_test = scalery.transform(np.array(y_test).reshape(-1, 1))
        y_train = scalery.transform(np.array(y_train).reshape(-1, 1))'''
        print(str(Sig)+'Signal')
        print(tick)
        model = RandomForestClassifier(n_estimators=1000, max_depth=5, max_features='sqrt', random_state=42)
        print(model)
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test,y_pred))
        #get the returns of the model
        y_pred[y_pred == 'BUY'] = 1
        y_pred[y_pred == 'SELL'] = -1
        y_pred[y_pred == 'NA'] = 0
        strat_returns = pd.Series(y_pred).shift(int(Sig))* X[str(Sig)+'day_returns'][X.index >= '2017-01-01'].values
        #get the returns of the perfect strategy
        y_test[y_test == 'BUY'] = 1
        y_test[y_test == 'SELL'] = -1
        y_test[y_test == 'NA'] = 0
        perf_returns = pd.Series(pd.Series(y_test).shift(int(Sig))* X[str(Sig)+'day_returns'][X.index >= '2017-01-01'].values).reset_index(drop=True)
        #returns of the market
        mkt_returns = pd.Series(X[str(Sig)+'day_returns'][X.index >= '2017-01-01'].values).reset_index(drop=True)
        #plot the result
        fig = plt.figure()
        ax = fig.add_subplot(111) #grabbing every nth row of returns to get the cum sum is what allows us mutiply to get 5 day returns every day
        ax.plot(perf_returns.iloc[::int(Sig)].expanding().sum(), color='b', label='Test '+str(Sig)+'day_Signal Perfect Strategy Returns') #need to only sum every 5th term..
        ax.plot(mkt_returns.iloc[::int(Sig)].expanding().sum(), color='r', label='Test'+str(Sig)+'day_Signal Market Returns')
        ax.plot(strat_returns.iloc[::int(Sig)].expanding().sum(), color='g', label='Test '+str(Sig)+'day_Signal Strategy Returns')
        plt.legend(loc='best')
        plt.title("Algo Test Returns")
        plt.show()
#save plots to compare - market return levels should be same but smoothed - ie. same max; strat and perf should ideally increase bc more trades to take advantage of more gains
#if doing application - can only make decision to buy/sell every 5 days.
#maybe pull in minute data in addition to daily ---> or what is better might be to take into account cost and model with hold, strong/weak/buy/sell

'''
print("Gbooster:")  #..36; cv gives 56 #terrible when scaled
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier()
model.fit(X_train, y_train)
model.score(X_test,y_test)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier #good when all vars are scaled
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
model.score(X_test,y_test)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))

print("ExtraTrees:")  #25
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))


model = Sequential()
model.add(Dense(128, input_dim=X_test.shape[1] ,init='uniform'))
model.add(Dense(64, init='uniform',activation='relu'))
model.add(Dense(8, init='uniform',activation='relu'))
model.add(Dense(1, init='uniform',activation='sigmoid'))
print(model.summary())

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae','mse','mape'])
nnet = model.fit(X_train,y_train, epochs = 100, batch_size=10)
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f" % (model.metrics_names[1],scores[1]))
print("\n%s: %.2f" % (model.metrics_names[2],scores[2]))
print("\n%s: %.2f%%" % (model.metrics_names[3],scores[3]*100))

from sklearn.neural_network import MLPClassifier
print("NNET:")
model = MLPClassifier(alpha=.00001,max_iter=5000) #57; cv .01
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))
'''