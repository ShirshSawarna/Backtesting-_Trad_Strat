#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


# In[32]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install yfinance')
get_ipython().system('pip install plotly')


# In[33]:





yf.pdr_override()


# In[34]:


#Getting the input of stock, Here Ticker means its symbol on Exchange ie- TSLA for Tesla Stock

stock = input("Enter a Stock Ticker Symbol: ")
print(stock)


# In[35]:


#Setting timeline and dates for intake of stock
startyear = 2021
startmonth = 1
startday = 1
start = dt.datetime(startyear,startmonth,startday)


now=dt.datetime.now()


# In[ ]:





# In[36]:




df=pdr.get_data_yahoo(stock,start,now)


# In[37]:


#Creating SMA 

ma=50

smaString="Sma_"+str(ma)
df[smaString]=df.iloc[:,4].rolling(window=ma).mean()

#removing NA values of SMA because SMA will be calculated only after 50 days

df = df.iloc[ma:]


# In[38]:


#Checking thet SMA is greater or lower than Clsong Price
numH=0
numL=0

for i in df.index:
    if(df["Adj Close"][i]>df[smaString][i]):
        print("The Close is Higher")
        numH=numH+1
    else: 
        print("The Close is Lower")
        numL+=1
        
print(str(numH))
print(str(numL))


# In[39]:


#Calculating Exponential Moving Average by creating a new column for every EMA

emasUsed=[3,5,8,10,12,15,30,35,40,45,50,60]
for x in emasUsed:
    ema=x
    df["Ema_"+str(ema)]=round(df.iloc[:,4].ewm(span=ema, adjust=False).mean(),2)


df=df.iloc[60:]




# In[40]:


#Checking for intersection of minimum value of short term ema and
#max value of long term ema and entering trade and checking Percentage profit loss in it


pos=0
num=0
percentchange=[]

for i in df.index:
    cmin=min(df["Ema_3"][i],df["Ema_5"][i],df["Ema_8"][i],df["Ema_10"][i],df["Ema_12"][i],df["Ema_15"][i],)
    cmax=max(df["Ema_30"][i],df["Ema_35"][i],df["Ema_40"][i],df["Ema_45"][i],df["Ema_50"][i],df["Ema_60"][i],)

    close=df["Adj Close"][i]
    if(cmin>cmax):
        print("Red White Blue")
        if(pos==0):
            bp=close
            pos=1
            print("Buying now at "+str(bp))


    elif(cmin<cmax):
        print("Blue White Red")
        if(pos==1):
            pos=0
            sp=close
            print("Selling now at "+str(sp))
            pc=(sp/bp-1)*100
            percentchange.append(pc)
if(num==df["Adj Close"].count()-1 and pos==1):
    pos=0
    sp=close
    print("Selling now at "+str(sp))
    pc=(sp/bp-1)*100
percentchange.append(pc)
    
num+=1

print(percentchange)


# In[41]:


#Calculating other general statistics to check our trade

gains=0
ng=0
losses=0
nl=0
totalR=1

for i in percentchange:
    if(i>0):
        gains+=i
        ng+=1
    else:
        losses+=i
        nl+=1
    totalR=totalR*((i/100)+1)

totalR=round((totalR-1)*100,2)

if(ng>0):
    avgGain=gains/ng
    maxR=str(max(percentchange))
else:
    avgGain=0
    maxR="undefined"

if(nl>0):
    avgLoss=losses/nl
    maxL=str(min(percentchange))
    ratio=str(-avgGain/avgLoss)
else:
    avgLoss=0
    maxL="undefined"
    ratio="inf"

if(ng>0 or nl>0):
    battingAvg=ng/(ng+nl)
else:
    battingAvg=0

print()
print("Results for "+ stock +" going back to "+str(df.index[0])+", Sample size: "+str(ng+nl)+" trades")
print("EMAs used: "+str(emasUsed))
print("Batting Avg: "+ str(battingAvg))
print("Gain/loss ratio: "+ ratio)
print("Average Gain: "+ str(avgGain))
print("Average Loss: "+ str(avgLoss))
print("Max Return: "+ maxR)
print("Max Loss: "+ maxL)
print("Total return over "+str(ng+nl)+ " trades: "+ str(totalR)+"%" )
#print("Example return Simulating "+str(n)+ " trades: "+ str(nReturn)+"%" )
print()


# In[42]:


#Creating Bollinger Bands

# Convert bars to DataFrame
df = pd.DataFrame(df)
df




# In[43]:


# calculate bollinger bands

# calculate sma
df['sma'] = df['close'].rolling(20).mean()

# calculate standard deviation
df['sd'] = df['close'].rolling(20).std()

# calculate lower band
df['lb'] = df['sma'] - 2 * df['sd']

# calculate upper band
df['ub'] = df['sma'] + 2 * df['sd']

df.dropna(inplace=True)
df


# In[44]:


# plotting close prices with bollinger bands
fig = px.line(df, x='time', y=['close', 'sma', 'lb', 'ub'])
fig


# In[45]:


# find signals

def find_signal(close, lower_band, upper_band):
    if close < lower_band:
        return 'buy'
    elif close > upper_band:
        return 'sell'
    
    
df['signal'] = np.vectorize(find_signal)(df['close'], df['lb'], df['ub'])

df


# In[46]:


# creating backtest and position classes

class Position:
    def __init__(self, open_datetime, open_price, order_type, volume, sl, tp):
        self.open_datetime = open_datetime
        self.open_price = open_price
        self.order_type = order_type
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.close_datetime = None
        self.close_price = None
        self.profit = None
        self.status = 'open'
        
    def close_position(self, close_datetime, close_price):
        self.close_datetime = close_datetime
        self.close_price = close_price
        self.profit = (self.close_price - self.open_price) * self.volume if self.order_type == 'buy'                                                                         else (self.open_price - self.close_price) * self.volume
        self.status = 'closed'
        
    def _asdict(self):
        return {
            'open_datetime': self.open_datetime,
            'open_price': self.open_price,
            'order_type': self.order_type,
            'volume': self.volume,
            'sl': self.sl,
            'tp': self.tp,
            'close_datetime': self.close_datetime,
            'close_price': self.close_price,
            'profit': self.profit,
            'status': self.status,
        }
        
        
class Strategy:
    def __init__(self, df, starting_balance, volume):
        self.starting_balance = starting_balance
        self.volume = volume
        self.positions = []
        self.data = df
        
    def get_positions_df(self):
        df = pd.DataFrame([position._asdict() for position in self.positions])
        df['pnl'] = df['profit'].cumsum() + self.starting_balance
        return df
        
    def add_position(self, position):
        self.positions.append(position)
        
    def trading_allowed(self):
        for pos in self.positions:
            if pos.status == 'open':
                return False
        
        return True
        
    def run(self):
        for i, data in self.data.iterrows():
            
            if data.signal == 'buy' and self.trading_allowed():
                sl = data.close - 3 * data.sd
                tp = data.close + 2 * data.sd
                self.add_position(Position(data.time, data.close, data.signal, self.volume, sl, tp))
                
            elif data.signal == 'sell' and self.trading_allowed():
                sl = data.close + 3 * data.sd
                tp = data.close - 2 * data.sd
                self.add_position(Position(data.time, data.close, data.signal, self.volume, sl, tp))
                
            for pos in self.positions:
                if pos.status == 'open':
                    if (pos.sl >= data.close and pos.order_type == 'buy'):
                        pos.close_position(data.time, pos.sl)
                    elif (pos.sl <= data.close and pos.order_type == 'sell'):
                        pos.close_position(data.time, pos.sl)
                    elif (pos.tp <= data.close and pos.order_type == 'buy'):
                        pos.close_position(data.time, pos.tp)
                    elif (pos.tp >= data.close and pos.order_type == 'sell'):
                        pos.close_position(data.time, pos.tp)
                        
        return self.get_positions_df()


# In[47]:


# run the backtest
bollinger_strategy = Strategy(df, 10000, 100000)
result = bollinger_strategy.run()

result


# In[48]:


# plotting close prices with bollinger bands
fig = px.line(df, x='time', y=['close', 'sma', 'lb', 'ub'])

# adding trades to plots
for i, position in result.iterrows():
    if position.status == 'closed':
        fig.add_shape(type="line",
            x0=position.open_datetime, y0=position.open_price, x1=position.close_datetime, y1=position.close_price,
            line=dict(
                color="green" if position.profit >= 0 else "red",
                width=3)
            )
fig


# In[49]:


# visualizing the results of the backtest
px.line(result, x='close_datetime', y='pnl')


# In[ ]:





# In[ ]:





# In[ ]:




