#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:24:36 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Stocks/') 

# load packages
# import warnings; warnings.simplefilter('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# default font
import matplotlib as mpl
mpl.rc('font',family='Helvetica')


# Load combined S&P 500 data set
sp = pd.read_csv('./data/raw/SP_5.4.2020.csv')
sp['date'] = pd.to_datetime(sp['Date'])
sp['close'] = sp['Close']
sp = sp[['date', 'close']]
sp['log close'] = np.log(sp['close'])
sp[::1000]


# add daily change, high price info to dataframe
sp_2 = pd.DataFrame({'date': sp['date'],
                     'close': sp['close'],
                     'daily change': round(sp.diff()['close'],4),
                     'daily change pct': round(100*sp.diff()['close']/sp['close'].shift(periods=1),3),
                     'high to date': sp['close'].cummax(),
                     'off from high': round(100*(sp['close'] - sp['close'].cummax()) / sp['close'].cummax(),3)})


# import data
# fract_pct_off = pd.read_csv('./data/fract_pct_off_1928_2020.csv')
# sp_changes = pd.read_csv('./data/sp_changes.csv')
# return_vs_dip_buy = pd.read_csv('./data/return_vs_dip_buy.csv')

# format date column to datetime
# sp_changes['date'] = pd.to_datetime(sp_changes['date'])

# select date range to filter
min_year = 1955
max_year = 2020
sp_3 = sp_2[(sp_2['date'] >= pd.Timestamp(min_year, 1, 1, 12)) & 
                  (sp_2['date'] <= pd.Timestamp(max_year+1, 1, 1, 12))]
sp_3[-50:]




### ADD NUMBER OF HIGHS COLUMN

# create running counter of number of record highs
sp_buys = sp_3.copy()
sp_buys['highs to date'] = 0
counter = 0
for index, row in sp_buys.iterrows():
    if sp_buys.at[index, 'off from high'] == 0:
        counter += 1
    sp_buys.at[index, 'highs to date'] = counter    
sp_buys[-60:]





### BUILD BUY TABLE

buy_dip = -10

sp_buys['buy'] = sp_buys['off from high'].apply(lambda x: True if x <= buy_dip else False)
sp_buys['shares bought'] = sp_buys['buy'].apply(lambda x: 1 if x == True else 0)
sp_buys['shares owned'] = sp_buys['shares bought'].cumsum()

sp_buys.loc[sp_buys['buy'] == True, 'cash spend'] = -sp_buys['close']
sp_buys.loc[sp_buys['buy'] == False, 'cash spend'] = 0

sp_buys['cash balance'] = sp_buys['cash spend'].cumsum()
sp_buys['market value'] = sp_buys['shares owned'] * sp_buys['close']
sp_buys['current gain'] = sp_buys['market value'] + sp_buys['cash balance']

cols = ['close', 'daily change', 'daily change pct', 'high to date', 'off from high', 'cash spend', 'cash balance', 'market value', 'current gain']
sp_buys[cols] = sp_buys[cols].round(2)

















return_vs_dip_buy = pd.DataFrame(columns=['dip', 'buy days', 'buy days (%)', 'return (%)'])

for dip in np.arange(-60,0.1,0.1):

    buy_dip = round(dip, 1)
    
    sp_buys['buy'] = sp_buys['off from high'].apply(lambda x: True if x <= buy_dip else False)
    sp_buys['shares bought'] = sp_buys['buy'].apply(lambda x: 1 if x == True else 0)
    sp_buys['shares owned'] = sp_buys['shares bought'].cumsum()
    
    sp_buys.loc[sp_buys['buy'] == True, 'cash spend'] = -sp_buys['close']
    sp_buys.loc[sp_buys['buy'] == False, 'cash spend'] = 0
    
    sp_buys['cash balance'] = sp_buys['cash spend'].cumsum()
    sp_buys['market value'] = sp_buys['shares owned'] * sp_buys['close']
    sp_buys['current gain'] = sp_buys['market value'] + sp_buys['cash balance']
    
    cols = ['close', 'daily change', 'daily change pct', 'high to date', 'off from high', 'cash spend', 'cash balance', 'market value', 'current gain']
    sp_buys[cols] = sp_buys[cols].round(2)
    
    if buy_dip % 10 == 0:
        sp_buys.to_csv('./data/buy tables/buy_table_dip=' + str(int(buy_dip)) + '.csv')
    
    # aggregate buy table for IRR calculation
    
    buy_days = sp_buys['shares owned'].iloc[-1]
    
    cash_outflows_yearly = sp_buys.groupby(sp_buys['date'].dt.year).sum()['cash spend']
    cash_inflow_final = pd.Series(sp_buys.iloc[-1]['market value'], index=[sp_3.iloc[-1]['date'].year])
    
    cash_flows = cash_outflows_yearly[:-1].append(pd.Series(cash_outflows_yearly.iloc[-1] + cash_inflow_final, index=[sp_3.iloc[-1]['date'].year]))
    # cash_flows = cash_outflows_yearly.append(cash_inflow_final)
    
    irr = np.irr(cash_flows)
    
    return_vs_dip_buy_temp = pd.DataFrame({'dip': buy_dip,
                                           'buy days': int(buy_days),
                                           'buy days (%)': 100*np.round(buy_days / len(sp_buys), 4),
                                           'return (%)': 100*np.round(irr, 4)}, index=[buy_dip])
    
    return_vs_dip_buy = return_vs_dip_buy.append(return_vs_dip_buy_temp)

return_vs_dip_buy = return_vs_dip_buy.reset_index().drop(columns=['index'])
return_vs_dip_buy.to_csv('./data/return_vs_dip_buy.csv')

































### Buy selectively within a dip


# identify days where price is n% off high
n = 10
sp_buys = sp_changes.copy()
sp_buys['dip'] = False

for index, row in sp_buys.iterrows():
    if sp_buys.at[index, 'off from high'] < -n:
        sp_buys.at[index, 'dip'] = True


# create running counter of number of record highs
sp_buys['highs to date'] = 0
counter = 0
for index, row in sp_buys.iterrows():
    if sp_buys.at[index, 'off from high'] == 0:
        counter += 1
    sp_buys.at[index, 'highs to date'] = counter    
sp_buys[:33]


# count up most frequent values for highs to date, get midpoint

highs_to_date_counts = sp_buys['highs to date'].value_counts().reset_index()
highs_to_date_counts.columns = ['number', 'counts']
divisions = 5
top_highs_to_date = highs_to_date_counts[:divisions]

# find day that high was reached
top_highs_date = pd.DataFrame(columns=['highs to date', 'date'])
for counter, i in enumerate(top_highs_to_date['number']):
    date_temp = sp_buys[sp_buys['highs to date'] == i].iloc[0]['date'].date()
    top_highs_date_temp = pd.DataFrame({'highs to date': i, 'date': date_temp}, index=[counter])
    top_highs_date = top_highs_date.append(top_highs_date_temp)


# top_highs_average_date = pd.DataFrame(columns=['highs to date', 'average date'])
# for counter, i in enumerate(top_highs_to_date['number']):
#     date_range = sp_buys[sp_buys['highs to date'] == i]
#     average_date = np.mean(date_range['date'])
#     top_highs_average_date_temp = pd.DataFrame({'highs to date': i, 'average date': average_date.date()}, index=[counter])
#     top_highs_average_date = top_highs_average_date.append(top_highs_average_date_temp)
    
# top_highs_average_date    



# linear scale
fig, axes = plt.subplots(1, 2, figsize = (18,7))
x = sp_changes['date']
y = sp_changes['close']
axes[0].plot(x, y, 'blue')
axes[0].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')

# log scale
axes[1].plot(x, y, 'blue')
axes[1].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_yscale('log')
axes[1].set_yticks(np.arange(500,3501,500))
axes[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# add line displaying top highs
for i in top_highs_date['date']:
    axes[0].axvline(x=i, linewidth=2, color='r', zorder=0)
    axes[1].axvline(x=i, linewidth=2, color='r', zorder=0)

# set axis tick label properties
plt.setp(axes[0].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[0].get_yticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# turn grid on
axes[0].grid(color=(.9, .9, .9))
axes[1].grid(color=(.9, .9, .9))

# ax.set_title('S&P 500: top highs', fontname='Helvetica', fontsize=22)

figure_name = './images/sp_top_highs.png'
plt.savefig(figure_name, dpi = 250)
plt.show()











pd.to_datetime(top_high.groupby('date').mean().date)

sp_buys['date'].iloc[:10]
np.mean(sp_buys['date'].iloc[:10])


# create new columns describing buys

# initialize new columns
sp_buys['shares bought'] = 0
sp_buys['total shares'] = np.nan
sp_buys['total spent'] = np.nan
sp_buys['avg buy price'] = np.nan
sp_buys['current return'] = np.nan
sp_buys['total spent'].iloc[0] = 0

# loop over rows, buying first dip and subsequent drops
for index, row in sp_buys.iterrows():
    
    # buy first day of n% dip
    if sp_buys.at[index, 'dip'] == True and sp_buys.at[index-1, 'dip'] == False:
        sp_buys.at[index, 'shares bought'] = 1 
        
    # buy when share price is below last buy price
    if len(sp_buys[sp_buys['shares bought'] == 1]) > 0:
        last_buy_price = sp_buys[sp_buys['shares bought'] == 1].iloc[-1]['close']
#     if last_buy_price:
#         if sp_buys.at[index, 'close'] < last_buy_price:
#             sp_buys.at[index, 'shares bought'] = 1

sp_buys.iloc[450:460]








# update shares bought, total shares, total spent, avg buy price, and current return columns        
sp_buys['total shares'] = sp_buys['shares bought'].cumsum()
# sp_buys['total spent'] = sp_buys['total shares']*


# sp_buys[sp_buys['dip'] == True].iloc[0]['shares bought'] = 1
# sp_buys['total spent'].iloc[0] = sp_buys['close'].iloc[0]
# sp_buys['avg buy price'].iloc[0] = sp_buys['close'].iloc[0]
# sp_buys['current return'].iloc[0] = 0
# sp_buys[:22]

# sp_buys[sp_buys['dip'] == True][:33]
# sp_buys.iloc[450:480]
sp_buys


 [426]:


# buy all subsequent days that are cheaper than average buy price
for index, row in sp_buys.iterrows():
    if sp_buys.at[index, 'dip'] == True:
        print(index, row)
    
#     # find last non-NaN element in last buy price and total spent columns
#     last_buy_price = sp_buys['close'].iloc[sp_buys['shares bought'][sp_buys['shares bought'] != 0].index[-1]]
    
#     # if current price is less than last buy price, buy another share
#     if row['close'] < last_buy_price:
#         sp_buys.at[index, 'shares bought'] = 1        

# populate shares owned column
sp_buys['shares owned'] = sp_buys['shares bought'].cumsum()
sp_buys


 [398]:


# populate total spent, averge buy price, and current return columns
for index, row in sp_buys[1:599].iterrows():
#     print(index)
#     sp_buys.at[index, 'total spent'] = sp_buys.at[index-1, 'total spent'] + (sp_buys.at[index, 'close'] * sp_buys.at[index, 'shares bought'])
    sp_buys.at[index, 'avg buy price'] = sp_buys.at[index, 'total spent'] / sp_buys.at[index, 'shares owned']
    sp_buys.at[index, 'current return'] = 100*(((sp_buys.at[index, 'shares owned'] * sp_buys.at[index, 'close']) / sp_buys.at[index, 'total spent']) - 1)
sp_buys[:199]


 [ ]:





 [ ]:





 [ ]:





 [ ]:


# find local extrema, create dataframes storing them
from scipy.signal import argrelextrema

order = 100
min_indices = argrelextrema(sp_changes['close'].values, np.less_equal, order=order)[0]
max_indices = argrelextrema(sp_changes['close'].values, np.greater_equal, order=order)[0]

min_dates = sp_changes.iloc[min_indices]['date']
max_dates = sp_changes.iloc[max_indices]['date']

mins = pd.DataFrame({'date': min_dates,
                     'close': sp_changes['close'][min_indices]})

maxs = pd.DataFrame({'date': max_dates,
                     'close': sp_changes['close'][max_indices]})


 [ ]:


### PLOT S&P 500 data with local extrema labeled

# linear scale
fig, axes = plt.subplots(1, 2, figsize = (18,7))
x = sp['date']
y = sp['close']
axes[0].plot(x, y, 'blue')
axes[0].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')

# log scale
import matplotlib
axes[1].plot(x, y, 'blue')
axes[1].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_yscale('log')
axes[1].set_yticks(np.arange(500,3501,500))
axes[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


# add labeled extrema
axes[0].scatter(mins['date'], mins['close'], c='r', edgecolor='k')
axes[0].scatter(maxs['date'], maxs['close'], c='g', edgecolor='k')
axes[1].scatter(mins['date'], mins['close'], c='r', edgecolor='k')
axes[1].scatter(maxs['date'], maxs['close'], c='g', edgecolor='k')

# set axis tick label properties
plt.setp(axes[0].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[0].get_yticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# turn grid on
axes[0].grid(color=(.9, .9, .9)); axes[0].set_axisbelow(True)
axes[1].grid(color=(.9, .9, .9)); axes[1].set_axisbelow(True)

figure_name = './images/sp_lin_log_extrema.png'

plt.savefig(figure_name, dpi = 250)

plt.show()


 [ ]:


sp_changes


 [ ]:


# find all dates where current price is x% ('drop') less than a past price within given period of time ('window')
num = 10000
drop = 0.40
window = 50

# loop over all closing prices
drops = pd.DataFrame(columns = ['start date', 'start price', 'end date', 'end price', 'drop pct'])
                               
for counter_i, close in enumerate(sp_changes['close'][:num]):
    
    # print status - year when searching early Jan
    if sp_changes['date'].iloc[counter_i].date().month == 1 and sp_changes['date'].iloc[counter_i].date().day < 4:
        print('***' + str(sp_changes['date'].iloc[counter_i].date().year) + '***')
    
    # loop over all prices after this
    for counter_j, close in enumerate(sp_changes['close'][:num]):
        
        # print dates within time window of peak x% higher
        if counter_j > counter_i and counter_j < window + counter_i and sp_changes['close'].iloc[counter_i] > (1 + drop)*sp_changes['close'].iloc[counter_j]:
                        
            # pull out important values
            start_date = sp_changes['date'].iloc[counter_i].date()
            start_price = sp_changes['close'].iloc[counter_i]
            end_date = sp_changes['date'].iloc[counter_j].date()
            end_price = sp_changes['close'].iloc[counter_j]
            drop_pct = round(100*(sp_changes['close'].iloc[counter_i]/sp_changes['close'].iloc[counter_j]-1),2)
            
#             # print them 
#             print(start_date, start_price, end_date, end_price, drop_pct)
            
            # create temporary dataframe
            drops_temp = pd.DataFrame({'start date': sp_changes['date'].iloc[counter_i].date(),
                                        'start price': sp_changes['close'].iloc[counter_i],
                                        'end date': sp_changes['date'].iloc[counter_j].date(),
                                        'end price': sp_changes['close'].iloc[counter_j],
                                        'drop pct': round(100*(sp_changes['close'].iloc[counter_i]/sp_changes['close'].iloc[counter_j]-1),2)}, index=[0])
            
            
            # add entry to growing dataframe, provided start and end dates are not already in there
            if start_date not in [item for item in drops['start date']] and end_date not in [item for item in drops['end date']]:
                drops = drops.append(drops_temp, ignore_index=True)


 [ ]:


drops


 [ ]:




