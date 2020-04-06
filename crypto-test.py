
# coding: utf-8

# In[53]:


from universal import tools
from universal import algos
import logging
# we would like to see algos progress
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.ERROR)


# In[1]:


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
plt.rcParams["figure.figsize"] = [20,5]


# In[2]:


intervals = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d']


# In[3]:


def get_ohlc_binance(symbol='ethbtc',start_date='30 days ago',interval='1d',indicators=None):
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


# In[4]:


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


# In[5]:


def get_ohlc(exchange='binance',symbol='ethbtc',start_date='30 days ago',interval='1day',indicators=None):
    ohlc = None
    if exchange=='binance':
        ohlc = get_ohlc_binance(symbol,start_date,interval,indicators)
    elif exchange=='tiingo':
        ohlc = get_ohlc_tiingo(symbol,start_date,interval,indicators)
    return ohlc


# In[6]:


def get_syms_binance(market='btc'):
    exchange_info = requests.get("https://api.binance.com/api/v1/exchangeInfo").json()
    return filter(None,[e['symbol'].lower() if e['quoteAsset']==market.upper() else '' for e in exchange_info['symbols']])


# In[7]:


def get_syms_tiingo(market='btc'):
    TOKEN = os.getenv('TIINGO_API_TOKEN')
    meta = requests.get("https://api.tiingo.com/tiingo/crypto?token={}".format(TOKEN)).json()
    return [item['ticker'] for item in meta if item["quoteCurrency"] == market]


# In[8]:


def get_syms(exchange='binance',market='btc'):
    syms = []
    if exchange=='binance':
        syms = get_syms_binance(market)
    elif exchange=='tiingo':
        syms = get_syms_tiingo(market)
    return syms


# In[26]:


def get_market(exchange='binance',market='usd',start_date='1 year ago',interval='1day',indicators=None):
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


# In[18]:


def get_markets(exchange='binance',markets=['usd','eth','btc'],start_date='1 year ago',interval='1day'):
    data = []
    for market in markets:
        df = get_market(exchange=exchange,market=market,start_date=start_date,interval=interval)
        if len(df):
            data.append(df)
    df = pd.concat(data,ignore_index=True)
    return df


# In[19]:


def to_sql(df,name='tiingo'):
    conn = sqlite3.connect(name+'.db')
    df.to_sql(name, conn, if_exists="replace")


# In[20]:


def read_sql(name='tiingo'):
    conn = sqlite3.connect(name+'.db')
    df = pd.read_sql_query("select * from {};".format(name), conn)    
    return df


# In[21]:


def to_pickle(df,name='tiingo'):
    df.to_pickle(name+'.pickle')


# In[22]:


def read_pickle(name='tiingo'):
    df = pd.read_pickle(name+'.pickle')
    return df


# In[272]:


df = get_markets(exchange='tiingo',markets=['btc'],start_date='1 year ago',interval='1d')
to_pickle(df,'btc_tiingo_1d')
df = read_pickle('btc_tiingo_1d')


# In[226]:


get_ipython().magic(u'ls *.pickle')


# In[273]:


df.symbol.unique()


# In[51]:


top = ['ethbtc','xrpbtc','bchbtc','ltcbtc','adabtc','neobtc','xlmbtc','eosbtc','miotabtc','dashbtc','xlmbtc','xmrbtc']


# In[184]:


df3 = df2.resample('1w').agg('last')


# In[277]:


#df2 = pd.pivot_table(df.loc[df['symbol'].isin(top)], values='close', index=['date'],columns=['symbol']).fillna(method='bfill')
df2 = pd.pivot_table(df, values='close', index=['date'],columns=['symbol']).fillna(method='bfill')


# In[278]:


(df2 / df2.iloc[0,:]).tail()


# In[299]:


#result = algos.OLMAR.run_combination(df2, window=[3,5,10,15], eps=10)
#result = algos.OLMAR.run_combination(df2, window=3, eps=10)
#algo = algos.DynamicCRP(n=52, min_history=8)
#result = algos.PAMR.run_combination(df2, eps=0.025)
#result.fee = 0.0025
test = df2.loc['2017-06-01':'2017-08-30']
#algo = algos.Anticor()
#algo = algos.BAH()
#algo = algos.BCRP()
#algo = algos.BNN()
#algo = algos.CORN()
#algo = algos.CRP()
#+ algo = algos.CWMR()
#algo = algos.EG()
#algo = algos.Kelly()
#algo = algos.OLMAR()
#algo = algos.ONS()
#algo = algos.PAMR()
#algo = algos.RMR()
#algo = algos.UP()
#algo = algos.DynamicCRP(n=52, min_history=8)
algo = algos.BestSoFar(n=10,metric='sharpe')
#algo = algos.BestSoFar()

list_result = algo.run(test)
list_result.fee = 0.0025
print(list_result.summary())
list_result.plot(figsize=(16,16))


# In[256]:


get_ipython().magic(u'ls *.pickle')


# In[281]:


df2.loc['2018':]


# In[380]:


ethbtc = pd.pivot_table(df[df.symbol=='ethbtc'], values='close', index=['date'],columns=['symbol'])
ethbtc = df[df.symbol=='ethbtc'].reset_index().close


# In[381]:


def getTEvents(gRaw,h):
    tEvents,sPos,sNeg = [],0,0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos,sNeg = max(0,sPos+diff.iloc[i]),min(0,sNeg+diff.iloc[i])
        if sNeg < -h:
            sNeg = 0; tEvents.append(i)
        elif sPos > h:
            sPos = 0; tEvents.append(i)
            
    return tEvents


# In[386]:


idx = getTEvents(df[df.symbol=='ethbtc'].reset_index().close,0.01)


# In[389]:


ethbtc.plot()


# In[388]:


ethbtc[idx].plot()


# In[ ]:




