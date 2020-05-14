# # Historical S&P 500 analysis

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Stocks/') 

# load packages
# import warnings; warnings.simplefilter('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
fract_pct_off = pd.read_csv('./data/fract_pct_off_1955_2020.csv')
# sp_changes = pd.read_csv('./data/sp_changes.csv')
# return_vs_dip_buy = pd.read_csv('./data/return_vs_dip_buy.csv')

# format date column to datetime
# sp_changes['date'] = pd.to_datetime(sp_changes['date'])

# select date range to filter
min_year = 1955
max_year = 2020
sp_3 = sp_2[(sp_2['date'] >= pd.Timestamp(min_year, 1, 1, 12)) & 
                  (sp_2['date'] <= pd.Timestamp(max_year+1, 1, 1, 12))]
sp_3






### CALCULATE ANNUALIZED RETURNS ACROSS ALL POSSIBLE TIME INTERVALS WITH MONTHLY SAMPLING ###
# from datetime import datetime as dt
# returns_interval = pd.DataFrame(columns=['time interval', 'start year', 'end year', 'start, end month', 'start price', 'end price', 'return (%)', 'annual return (%)'])
# returns_across_time_intervals = pd.DataFrame(columns=['time interval', 'avg return (%)', 'avg annual return (%)', 'annual return stdev (%)'])

# total_time = int((sp_3['date'].iloc[-1] - sp_3['date'].iloc[0]).days / 365)
# now = dt.now()

# for i in range(2, total_time, 1):
#     year_span = i - 1
    
#     for j in range(1928, 2020 - year_span + 1, 1):
        
#         # get start and end price over given time interval
#         start_year = j
#         end_year = j + year_span
        
#         print(year_span, start_year, end_year)
#         print('***')
        
#         for month in np.arange(1,13,1):
#             print(month)
#             start_date = pd.Timestamp(start_year, month, 1, 0)
#             end_date = pd.Timestamp(end_year, month, 1, 0)
            
#             start_price = sp_2[sp_2['date'] >= start_date].iloc[0]['close']
            
#             if now > end_date:
#                 end_price = sp_2[sp_2['date'] >= end_date].iloc[0]['close']
        
                
#                 # append return data to growing dataframe 
#                 returns_interval_temp = pd.DataFrame({'time interval': year_span,
#                                                       'start year': start_year,
#                                                       'end year': end_year,
#                                                       'start, end month': month,
#                                                       'start price': start_price,
#                                                       'end price': end_price,
#                                                       'return (%)': 100*((end_price / start_price) - 1),
#                                                       'annual return (%)': 100*((end_price / start_price) ** (1/year_span) - 1)
#                                                       }, index=[0])
#                 returns_interval = returns_interval.append(returns_interval_temp, ignore_index = True)

#     # append statistical data to growing dataframe
#     returns_across_time_intervals_temp = pd.DataFrame({'time interval': year_span,
#                                                         'avg return (%)': round(np.nanmean(returns_interval['return (%)']),3),
#                                                         'avg annual return (%)': round(np.nanmean(returns_interval['annual return (%)']), 3),
#                                                         'annual return stdev (%)': round(np.std(returns_interval['annual return (%)']), 3),
#                                                         }, index=[0])

#     returns_across_time_intervals = returns_across_time_intervals.append(returns_across_time_intervals_temp, ignore_index = True)
        
    
# returns_interval.to_csv('./data/5 - returns across time intervals/returns_across_time_intervals_full_'+ 'monthly.csv', index=False)
# returns_across_time_intervals.to_csv('./data/5 - returns across time intervals/returns_across_time_intervals_' + 'monthly.csv', index=False)



### PLOT HISTORICAL RETURNS BY INVESTMENT HORIZON ###

returns_interval = pd.read_csv('./data/5 - returns across time intervals/returns_across_time_intervals_full_monthly.csv')

# # calculate statistical moments of returns across time intervals
# returns_1y = returns_interval[returns_interval['time interval'] == 1]['annual return (%)']
# returns_10y = returns_interval[returns_interval['time interval'] == 10]['annual return (%)']
# returns_30y = returns_interval[returns_interval['time interval'] == 30]['annual return (%)']

# one_year = [round(np.mean(returns_1y), 2), round(np.median(returns_1y), 2), round(np.std(returns_1y), 2)]
# ten_year = [round(np.mean(returns_10y), 2), round(np.median(returns_10y), 2), round(np.std(returns_10y), 2)]
# thirty_year = [round(np.mean(returns_30y), 2), round(np.median(returns_30y), 2), round(np.std(returns_30y), 2)]

# returns_summary = pd.DataFrame({' ': ['Mean', 'Median', 'St. dev.'],
#                                 'One year': one_year,
#                                 'Ten years': ten_year,
#                                 'Thirty years': thirty_year})

# returns_summary.to_csv('./data/5 - returns across time intervals/return_summary.csv', index=False)



# select date range to filter
min_year = 1955
max_year = 2020
returns_interval_filtered = returns_interval[(returns_interval['start year'] >= min_year) & 
            (returns_interval['start year'] <= max_year)]

total_time = max_year - min_year

# create seaborn box + strip plot
import seaborn as sns
fig, ax = plt.subplots(1, 1, figsize = (20, 20))

ax = sns.boxplot(x = 'time interval', y = 'annual return (%)', data = returns_interval_filtered, 
                 showfliers = False, order = list(set(returns_interval['time interval'])), linewidth = 1)
ax = sns.stripplot(x = 'time interval', y = 'annual return (%)', data = returns_interval_filtered,
                 order = list(set(returns_interval['time interval'])), jitter = 0.25, size = 5,
                 linewidth = 1, edgecolor = 'black', alpha = 0.25)

ax.axhline(y=0, linestyle=':', linewidth=3, color='grey')

# set axis properties
plt.xticks(fontname = 'Helvetica', fontsize = 42)
plt.yticks(fontname = 'Helvetica', fontsize = 42)

import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.xlabel('Time horizon (years)', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Annual return (%)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(0, 14); ax.yaxis.labelpad = 25
# ax.set_ylim(-40, 50); ax.yaxis.labelpad = 25

ax.set_xlim(-1, total_time); ax.xaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3)

# turn grid on
plt.grid(color=(.75, .75, .75))
plt.grid(color=(.75, .75, .75))

figure_name = './images/3 - returns over time intervals/returns_vs_interval_' + str(min_year) + '_' + str(max_year) + '.png'

plt.tight_layout()
# plt.savefig(figure_name, dpi = 150)
plt.show()












