
import urllib.request
from urllib import request
import pandas as pd

DailyURL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&outputsize={OUTPUTSIZE}&apikey={KEY}&datatype=csv"
IntradayURL = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={SYMBOL}&interval={INTERVAL}&outputsize={OUTPUTSIZE}&apikey={KEY}&datatype=csv"
API_KEY = "************"

def get_daily(symbol,outputsize):
    #output size can be full or compact
    with urllib.request.urlopen(DailyURL.format(OUTPUTSIZE=outputsize, KEY=API_KEY, SYMBOL=symbol)) as req:
        data = pd.read_csv(req)
    return data

def get_intraday(symbol,interval,outputsize):
    #output size can be full or compact
    with urllib.request.urlopen(IntradayURL.format(OUTPUTSIZE=outputsize,INTERVAL = interval, KEY=API_KEY, SYMBOL=symbol)) as req:
        data = pd.read_csv(req)
    return data






