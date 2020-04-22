# # Historical S&P 500 analysis

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Stocks/') 

# load packages
# import warnings; warnings.simplefilter('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# import data
fract_pct_off = pd.read_csv('./data/fract_pct_off.csv')
sp_diffs = pd.read_csv('./data/sp_diffs.csv')
return_vs_dip_buy = pd.read_csv('./data/return_vs_dip_buy.csv')

# format date column to datetime
sp_diffs['date'] = pd.to_datetime(sp_diffs['date'])

# select years
year = 1986
sp_diffs = sp_diffs[sp_diffs['date'].dt.year >= year]


### Calculate return vs. buy strategy


# define function to calculate price paid, shares purchased, avg share price, roi for given buy dates
def calc_return(dip, buy_dates):
    
    # sum up share prices to get price paid
    paid = int(buy_dates['close'].sum())
    
    # sum up number of shares bought to get average share price
    shares = len(buy_dates)
    avg_share_price = round(paid/shares, 2)
    
    # take current (last available) price as market value per share
    current_price = sp_diffs['close'].iloc[-1]
    
    # calculate total ROI and annualized ROI
    roi = 100*round((current_price - avg_share_price) / avg_share_price, 3)
    roi_annual = round(roi / (sp_diffs['date'].iloc[-1].year - sp_diffs['date'].iloc[0].year), 2)
    
    # create dataframe summarizing results
    out = pd.DataFrame({'dip': dip,
                        'paid': paid, 
                        'shares': shares,
                        'buy day fraction': 100*shares/len(sp_diffs),
                        'avg share price': avg_share_price, 
                        'current price': current_price, 
                        'ROI': roi, 
                        'ROI/y': roi_annual}, index=[0])
    
    return out


### Buy throughout a dip

# baseline: buy every day
calc_return(0, sp_diffs)

# buy when discount exceeds n%
n = 15
discount_n = sp_diffs[sp_diffs['off from high'] < -n]
calc_return(n, discount_n)



# # create return table: return vs. buy dip %
# return_table = pd.DataFrame(columns=['dip', 'paid', 'shares', 'buy day fraction', 'avg share price', 'current price', 'ROI', 'ROI/y'])
# for i in np.arange(0, 56, 0.1):
#     discount_n = sp_diffs[sp_diffs['off from high'] <= -i]
#     return_temp = calc_return(i, discount_n)
#     return_table = return_table.append(return_temp)


# # format, save return vs. dip buy table
# return_vs_dip_buy = return_table.round({'dip': 1, 'ROI': 1})
# return_vs_dip_buy.to_csv('./data/return_vs_dip_buy.csv', index=False)



# print summary table 
return_vs_dip_buy_summary = return_vs_dip_buy[::50][['dip', 'buy day fraction', 'ROI/y']].round({'buy day fraction': 2}).astype({'dip': int}).reset_index().drop(columns='index')
return_vs_dip_buy_summary.columns = ['Percent off (%)', 'Fraction of days (%)', 'Annual return (%)']
# return_vs_dip_buy_summary.style.hide_index()


# plot line of return vs. dip buy

### PLOT S&P 500 data
fig, ax = plt.subplots(1, 1, figsize = (7,7))
ax2 = ax.twinx()

# buy opportunities
x = -fract_pct_off['percent off']
y = fract_pct_off['fraction of days']
ax.plot(-x, y, 'blue', linewidth=3, zorder=99)
ax.set_xlabel('Down from high (%)', fontsize = 22, fontname = 'Helvetica', fontweight = 'bold')
ax.set_ylabel('Fraction of closes (%)', fontsize = 22, c = 'blue', fontname = 'Helvetica', fontweight = 'bold')
# axes[0].set_ylim(-1,105)

# returns
x2 = return_vs_dip_buy['dip']
y2 = return_vs_dip_buy['ROI/y']
ax2.plot(x2, y2, 'green', linewidth=3, zorder=15)
ax2.set_xlabel('Buy when below (%)', fontsize = 22, fontname = 'Helvetica', fontweight = 'bold')
ax2.set_ylabel('Annual return (%)', fontsize = 22, c = 'darkgreen', fontname = 'Helvetica', fontweight = 'bold', labelpad= 30, rotation=270)
# axes[0].set_ylim(-1,105)

# set axis tick label properties
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=18, fontname = 'Helvetica')
plt.setp(ax.get_yticklabels(), fontsize=18, fontname = 'Helvetica')
plt.setp(ax2.get_yticklabels(), fontsize=18, fontname = 'Helvetica', rotation=0)

# turn grid on
ax.grid(color=(.9, .9, .9)); 

figure_name = './images/roi_vs_dip_buy_since_' + str(year) + '.png'

plt.savefig(figure_name, dpi = 250)
plt.subplots_adjust(wspace=0.25)
plt.show()


# create visual display of buying dips

# get buy dates
n = 10
sp_with_buy_dates = sp_diffs.copy()
sp_with_buy_dates['buy'] = sp_diffs['off from high'] < -n

# linear scale
fig, axes = plt.subplots(1, 2, figsize = (18,7))
x = sp_diffs['date']
y = sp_diffs['close']
axes[0].plot(x, y, 'blue')
axes[0].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].fill_between(x=sp_with_buy_dates['date'], y1=sp_with_buy_dates['close'], y2=0, facecolor='r', where=sp_with_buy_dates['buy']==True)

# log scale
axes[1].plot(x, y, 'blue')
axes[1].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_yscale('log')
axes[1].set_yticks(np.arange(500,3501,500))
axes[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axes[1].fill_between(x=sp_with_buy_dates['date'], y1=sp_with_buy_dates['close'], y2=5, facecolor='r', where=sp_with_buy_dates['buy']==True)

# set axis tick label properties
plt.setp(axes[0].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[0].get_yticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# turn grid on
axes[0].grid(color=(.9, .9, .9))
axes[1].grid(color=(.9, .9, .9))

axes[0].set_title('S&P 500:  ' + str(n) + '% off', fontname='Helvetica', fontsize=22)
axes[1].set_title('S&P 500:  ' + str(n) + '% off', fontname='Helvetica', fontsize=22)

figure_name = './images/sp_lin_log_buy_dip_' + str(n) + '.png'
plt.savefig(figure_name, dpi = 250)
plt.show()



# create visual display of buying dips for various cases

# get buy dates
start = 10
stop = 45
step = 5
sp_with_buy_dates = sp_diffs.copy()
sp_with_buy_dates['buy'] = sp_diffs['off from high'] < -n

# linear scale
nrow = 2
ncol = 4
fig, axes = plt.subplots(nrow, ncol, figsize = (25,15))

# loop over dip buy options
for counter, i in enumerate(np.arange(start, stop+1, step)):

    # get buy dates
    sp_with_buy_dates['buy'] = sp_diffs['off from high'] < -i

    x = sp_diffs['date']
    y = sp_diffs['close']
    axes[counter//ncol, counter%ncol].plot(x, y, 'blue')
    axes[counter//ncol, counter%ncol].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
    axes[counter//ncol, counter%ncol].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
    axes[counter//ncol, counter%ncol].fill_between(x=sp_with_buy_dates['date'], y1=sp_with_buy_dates['close'], y2=0, facecolor='r', where=sp_with_buy_dates['buy']==True)

    # set axis tick label properties
    plt.setp(axes[counter//ncol, counter%ncol].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
    plt.setp(axes[counter//ncol, counter%ncol].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

    # turn grid on
    axes[counter//ncol, counter%ncol].grid(color=(.9, .9, .9)); axes[counter//ncol, counter%ncol].set_axisbelow(True)
    axes[counter//ncol, counter%ncol].set_title('S&P 500:  ' + str(i) + '% off', fontname='Helvetica', fontsize=22)

plt.gcf().subplots_adjust(bottom=0.15, left = 0.035, right = 0.985)
plt.subplots_adjust(wspace=0.35, hspace=0.55)
figure_name = './images/sp_lin_buy_dip_composite_2.png'
# plt.savefig(figure_name, dpi = 250)
plt.show()


# ### Buy selectively within a dip


# identify days where price is n% off high
n = 10
sp_buys = sp_diffs.copy()
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
x = sp_diffs['date']
y = sp_diffs['close']
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
min_indices = argrelextrema(sp_diffs['close'].values, np.less_equal, order=order)[0]
max_indices = argrelextrema(sp_diffs['close'].values, np.greater_equal, order=order)[0]

min_dates = sp_diffs.iloc[min_indices]['date']
max_dates = sp_diffs.iloc[max_indices]['date']

mins = pd.DataFrame({'date': min_dates,
                     'close': sp_diffs['close'][min_indices]})

maxs = pd.DataFrame({'date': max_dates,
                     'close': sp_diffs['close'][max_indices]})


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


sp_diffs


 [ ]:


# find all dates where current price is x% ('drop') less than a past price within given period of time ('window')
num = 10000
drop = 0.40
window = 50

# loop over all closing prices
drops = pd.DataFrame(columns = ['start date', 'start price', 'end date', 'end price', 'drop pct'])
                               
for counter_i, close in enumerate(sp_diffs['close'][:num]):
    
    # print status - year when searching early Jan
    if sp_diffs['date'].iloc[counter_i].date().month == 1 and sp_diffs['date'].iloc[counter_i].date().day < 4:
        print('***' + str(sp_diffs['date'].iloc[counter_i].date().year) + '***')
    
    # loop over all prices after this
    for counter_j, close in enumerate(sp_diffs['close'][:num]):
        
        # print dates within time window of peak x% higher
        if counter_j > counter_i and counter_j < window + counter_i and sp_diffs['close'].iloc[counter_i] > (1 + drop)*sp_diffs['close'].iloc[counter_j]:
                        
            # pull out important values
            start_date = sp_diffs['date'].iloc[counter_i].date()
            start_price = sp_diffs['close'].iloc[counter_i]
            end_date = sp_diffs['date'].iloc[counter_j].date()
            end_price = sp_diffs['close'].iloc[counter_j]
            drop_pct = round(100*(sp_diffs['close'].iloc[counter_i]/sp_diffs['close'].iloc[counter_j]-1),2)
            
#             # print them 
#             print(start_date, start_price, end_date, end_price, drop_pct)
            
            # create temporary dataframe
            drops_temp = pd.DataFrame({'start date': sp_diffs['date'].iloc[counter_i].date(),
                                        'start price': sp_diffs['close'].iloc[counter_i],
                                        'end date': sp_diffs['date'].iloc[counter_j].date(),
                                        'end price': sp_diffs['close'].iloc[counter_j],
                                        'drop pct': round(100*(sp_diffs['close'].iloc[counter_i]/sp_diffs['close'].iloc[counter_j]-1),2)}, index=[0])
            
            
            # add entry to growing dataframe, provided start and end dates are not already in there
            if start_date not in [item for item in drops['start date']] and end_date not in [item for item in drops['end date']]:
                drops = drops.append(drops_temp, ignore_index=True)


 [ ]:


drops


 [ ]:




