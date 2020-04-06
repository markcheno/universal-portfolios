import os
import talib
import sqlite3
import requests
import numpy as np
import pandas as pd
import dateparser as dp
from binance.client import Client

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,8]


def get_ohlc_binance(symbol='ethbtc',start_date='30 days ago',interval='1d',indicators=None):
    assert interval in ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d']
    client = Client('','')
    ohlc = client.get_historical_klines(symbol.upper(),interval,start_date,)         
    if ohlc and len(ohlc) > 0:
        date = [pd.to_datetime(i[0], unit='ms') for i in ohlc]
        _open = [float(i[1]) for i in ohlc]
        _high = [float(i[2]) for i in ohlc]
        _low = [float(i[3]) for i in ohlc]
        _close = [float(i[4]) for i in ohlc]
        _volume = [float(i[5]) for i in ohlc]
        sym = [symbol.lower()]*len(_close)
        df = pd.DataFrame({'symbol':sym,'date':date,'open':_open,'high':_high,'low':_low,'close':_close,'volume':_volume})
        if indicators:
            df = indicators(df)
        return df
    return None

def get_ohlc_tiingo(symbol='ethbtc',start_date='30 days ago',interval='1d',indicators=None):
    # @param:interval This allows you to set the frequency in which you want data resampled. 
    # For example "1hour" would return the data where OHLC is calculated on an hourly schedule. 
    # The minimum value is "1min". 
    # Units in minutes (min), hours (hour), and days (day) are accepted.
    # Format is # + (min/hour/day)
    if 'm' in interval and not 'min' in interval:
        interval = interval.replace('m','min')
    if 'h' in interval and not 'hour' in interval:
        interval = interval.replace('h','hour')
    if 'd' in interval and not 'day' in interval:
        interval = interval.replace('d','day')        
    assert('min' in interval or 'hour' in interval or 'day' in interval)
    
    TOKEN = os.getenv('TIINGO_API_TOKEN')
    startDate = dp.parse(start_date).strftime('%Y-%m-%d')
    # https://api.tiingo.com/docs/crypto
    url = "https://api.tiingo.com/tiingo/crypto/prices?tickers={}&resampleFreq={}&startDate={}&token={}".format(symbol,interval,startDate,TOKEN)
    data = requests.get(url).json()
    d,o,h,l,c,v = [],[],[],[],[],[]
    if len(data):
        for bar in data[0]['priceData']:
            d.append(dp.parse(bar['date']))
            o.append(bar['open'])
            h.append(bar['high'])
            l.append(bar['low'])
            c.append(bar['close'])
            v.append(bar['volumeNotional']) # close * volume in quote currency
            
        df = pd.DataFrame({'symbol':symbol, 'date': d, 'open':o, 'high':h, 'low':l, 'close':c, 'volume':v})
        if indicators:
            df = indicators(df)

        return df
    return None

def get_ohlc(exchange='binance',symbol='ethbtc',start_date='30 days ago',interval='1day',indicators=None):
    ohlc = None
    if exchange=='binance':
        ohlc = get_ohlc_binance(symbol,start_date,interval,indicators)
    elif exchange=='tiingo':
        ohlc = get_ohlc_tiingo(symbol,start_date,interval,indicators)
    return ohlc

def get_syms_binance(market='btc'):
    exchange_info = requests.get("https://api.binance.com/api/v1/exchangeInfo").json()
    return filter(None,[e['symbol'].lower() if e['quoteAsset']==market.upper() else '' for e in exchange_info['symbols']])

def get_syms_tiingo(market='btc'):
    TOKEN = os.getenv('TIINGO_API_TOKEN')
    meta = requests.get("https://api.tiingo.com/tiingo/crypto?token={}".format(TOKEN)).json()
    return [item['ticker'] for item in meta if item["quoteCurrency"] == market]

def get_syms(exchange='binance',market='btc'):
    syms = []
    if exchange=='binance':
        syms = get_syms_binance(market)
    elif exchange=='tiingo':
        syms = get_syms_tiingo(market)
    return syms

def get_market(exchange='binance',market='usd',start_date='1 year ago',interval='1d',indicators=None):
    syms = get_syms(exchange,market)
    data = []
    for sym in syms:
        df = get_ohlc(exchange,sym,start_date,interval,indicators)
        if df is not None:
            df['market'] = market
            data.append(df)
    df = pd.concat(data,ignore_index=True)
    df['date'] = df.date.apply(lambda d: d.replace(tzinfo=None))
    return df

def get_markets(exchange='binance',markets=['usd','eth','btc'],start_date='1 year ago',interval='1d'):
    data = []
    for market in markets:
        df = get_market(exchange=exchange,market=market,start_date=start_date,interval=interval)
        if len(df):
            data.append(df)
    df = pd.concat(data,ignore_index=True)
    return df

def get_closes(df):
    return pd.pivot_table(df, values='close', index=['date'],columns=['symbol']).fillna(method='bfill')

def to_sql(df,name):
    conn = sqlite3.connect(name+'.db')
    df.to_sql(name, conn, if_exists="replace")

def read_sql(name):
    conn = sqlite3.connect(name+'.db')
    df = pd.read_sql_query("select * from {};".format(name), conn)    
    return df

def to_pickle(df,name):
    df.to_pickle(name+'.pickle')

def read_pickle(name):
    df = pd.read_pickle(name+'.pickle')
    return df
