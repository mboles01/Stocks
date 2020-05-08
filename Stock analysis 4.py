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
sp_3


### BUILD BUY TABLE: BUY WHENEVER DIP EXCEEDS THRESHOLD

return_vs_dip_buy = pd.DataFrame(columns=['dip', 'buy days', 'buy days (%)', 'return (%)'])

sp_buys = sp_3.copy()

for dip in np.arange(-60,0.1,0.1):

    buy_dip = round(dip, 1)
    
    # buy = True whenever dip exceeds threshold
    sp_buys['buy'] = sp_buys['off from high'].apply(lambda x: True if x <= buy_dip else False)
    sp_buys['shares bought'] = sp_buys['buy'].apply(lambda x: 1 if x == True else 0)
    sp_buys['shares owned'] = sp_buys['shares bought'].cumsum()
    
    # when buying, cash spent is share price (buy one share)
    sp_buys.loc[sp_buys['buy'] == True, 'cash spend'] = -sp_buys['close']
    sp_buys.loc[sp_buys['buy'] == False, 'cash spend'] = 0
    
    # add summary columns: cash balance, market value of shares, and current gain/loss
    sp_buys['cash balance'] = sp_buys['cash spend'].cumsum()
    sp_buys['market value'] = sp_buys['shares owned'] * sp_buys['close']
    sp_buys['current gain'] = sp_buys['market value'] + sp_buys['cash balance']
    
    cols = ['close', 'daily change', 'daily change pct', 'high to date', 'off from high', 'cash spend', 'cash balance', 'market value', 'current gain']
    sp_buys[cols] = sp_buys[cols].round(2)
    
    # save buy tables for dip increments of 10%
    if buy_dip % 10 == 0:
        sp_buys.to_csv('./data/buy tables/buy_table_dip=' + str(int(buy_dip)) + '.csv', index=False)
    
    
    # aggregate buy table to annual cash inflows/outflows for IRR calculation
    cash_outflows_yearly = sp_buys.groupby(sp_buys['date'].dt.year).sum()['cash spend']
    cash_inflow_final = pd.Series(sp_buys.iloc[-1]['market value'], index=[sp_3.iloc[-1]['date'].year])


    # aggregate monthly instead of yearly 
    sp_buys.insert(0, column='year', value=sp_buys['date'].dt.year)
    sp_buys.insert(0, column='month', value=sp_buys['date'].dt.month)
    sp_buys.insert(0, column='day', value=sp_buys['date'].dt.day)
    cash_outflows_monthly = sp_buys.groupby(by=[sp_buys['year'], sp_buys['month']]).sum()['cash spend']
    
    # no aggregation - can I calculate IRR from daily data?
    asdf
    
    

    # last cash flow is sum of last (interval - year/month)'s stock purchase plus implicit sale of entire position    
    cash_flows_yearly_all = cash_outflows_yearly[:-1].append(pd.Series(cash_outflows_yearly.iloc[-1] + cash_inflow_final, index=[sp_3.iloc[-1]['date'].year]))
    cash_flows_monthly_all = cash_outflows_monthly[:-1].append(pd.Series(cash_outflows_monthly.iloc[-1] + cash_inflow_final, index=[sp_3.iloc[-1]['date'].year]))
    
    # calculate internal rate (IRR) of return of this cash flow schedule
    irr_yearly = np.irr(cash_flows_yearly_all)  # bigger bc assumes May 1 value recovered on Jan 1
    irr_monthly = np.irr(cash_flows_monthly_all)*12 # smaller bc May 1 value recovered on May 1 (it's correct)

    
    # build growing table of returns vs. threshold for dip buying
    return_vs_dip_buy_temp = pd.DataFrame({'dip': buy_dip,
                                           'buy days': int(sp_buys['shares owned'].iloc[-1]),
                                           'buy days (%)': 100*np.round(sp_buys['shares owned'].iloc[-1] / len(sp_buys), 4),
                                           'return (%)': 100*np.round(irr, 4)}, index=[buy_dip])
    
    return_vs_dip_buy = return_vs_dip_buy.append(return_vs_dip_buy_temp)

# # save completed table
# return_vs_dip_buy = return_vs_dip_buy.reset_index().drop(columns=['index'])
# return_vs_dip_buy.to_csv('./data/return_vs_dip_buy.csv', index=False)







### PLOT LINE OF RETURN VS. DIP BUY

return_vs_dip_buy = pd.read_csv('./data/6 - return vs dip buy/return_vs_dip_buy.csv')

fig, ax = plt.subplots(1, 1, figsize = (8,7))
ax2 = ax.twinx()

# buy opportunities
x = return_vs_dip_buy['dip']
y = return_vs_dip_buy['buy days (%)']
ax.plot(-x, y, 'blue', linewidth=3, zorder=99)
ax.set_xlabel('Down from high (%)', fontsize = 22, fontname = 'Helvetica', fontweight = 'bold')
ax.set_ylabel('Fraction of closes (%)', fontsize = 22, c = 'blue', fontname = 'Helvetica', fontweight = 'bold')
# axes[0].set_ylim(-1,105)

# returns
x2 = return_vs_dip_buy['dip']
y2 = return_vs_dip_buy['return (%)']
ax2.plot(-x2, y2, 'green', linewidth=3, zorder=15)
ax2.set_xlabel('Buy when below (%)', fontsize = 22, fontname = 'Helvetica', fontweight = 'bold')
ax2.set_ylabel('Annual return (%)', fontsize = 22, c = 'darkgreen', fontname = 'Helvetica', fontweight = 'bold', labelpad= 30, rotation=270)
# axes[0].set_ylim(-1,105)

# set axis tick label properties
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=18, fontname = 'Helvetica')
plt.setp(ax.get_yticklabels(), fontsize=18, fontname = 'Helvetica')
plt.setp(ax2.get_yticklabels(), fontsize=18, fontname = 'Helvetica', rotation=0)

# turn grid on
ax.grid(color=(.9, .9, .9))
# ax.grid(color=(1, 1, 1))
ax2.grid(False)

figure_name = './images/7 - return vs. dip buy/return_vs_dip_buy_' + str(min_year) + '_' + str(max_year) + '.png'

# plt.subplots_adjust(wspace=0.25)
plt.tight_layout()
plt.savefig(figure_name, dpi = 250)
plt.show()



# ### CREATE VISUAL DISPLAY OF BUYING DIPS OF GIVEN MINIMUM DOWN %

# # get buy dates
# n = 10
# sp_with_buy_dates = sp_changes.copy()
# sp_with_buy_dates['buy'] = sp_changes['off from high'] < -n

# # linear scale
# fig, axes = plt.subplots(1, 2, figsize = (18,7))
# x = sp_changes['date']
# y = sp_changes['close']
# axes[0].plot(x, y, 'blue')
# axes[0].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
# axes[0].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
# axes[0].fill_between(x=sp_with_buy_dates['date'], y1=sp_with_buy_dates['close'], y2=0, facecolor='r', where=sp_with_buy_dates['buy']==True)

# # log scale
# axes[1].plot(x, y, 'blue')
# axes[1].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
# axes[1].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
# axes[1].set_yscale('log')
# axes[1].set_yticks(np.arange(500,3501,500))
# axes[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# axes[1].fill_between(x=sp_with_buy_dates['date'], y1=sp_with_buy_dates['close'], y2=5, facecolor='r', where=sp_with_buy_dates['buy']==True)

# # set axis tick label properties
# plt.setp(axes[0].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
# plt.setp(axes[1].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
# plt.setp(axes[0].get_yticklabels(), fontsize=14, fontname = 'Helvetica')
# plt.setp(axes[1].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# # turn grid on
# axes[0].grid(color=(.9, .9, .9))
# axes[1].grid(color=(.9, .9, .9))

# axes[0].set_title('S&P 500:  ' + str(n) + '% off', fontname='Helvetica', fontsize=22)
# axes[1].set_title('S&P 500:  ' + str(n) + '% off', fontname='Helvetica', fontsize=22)

# figure_name = './images/sp_lin_log_buy_dip_since_' + str(min_year) + '.png'
# # plt.savefig(figure_name, dpi = 250)
# plt.show()



### CREATE VISUAL DISPLAY OF BUYING DIPS FOR VARIOUS MINIMUM DIPS

# get buy dates
start = 10
stop = 45
step = 5
sp_with_buy_dates = sp_3.copy()
# sp_with_buy_dates['buy'] = sp_3['off from high'] < -n

# linear scale
nrow = 2
ncol = 4
fig, axes = plt.subplots(nrow, ncol, figsize = (25,15))

# loop over dip buy options
for counter, i in enumerate(np.arange(start, stop+1, step)):

    # get buy dates
    sp_with_buy_dates['buy'] = sp_3['off from high'] < -i

    x = sp_3['date']
    y = sp_3['close']
    axes[counter//ncol, counter%ncol].plot(x, y, 'blue', label='S&P 500')
    axes[counter//ncol, counter%ncol].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
    axes[counter//ncol, counter%ncol].set_yscale('log')
    axes[counter//ncol, counter%ncol].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
    axes[counter//ncol, counter%ncol].fill_between(x=sp_with_buy_dates['date'], y1=sp_with_buy_dates['close'], label = str(i) + '% off', y2=20, facecolor='r', where=sp_with_buy_dates['buy']==True)

    # set axis tick label properties
    plt.setp(axes[counter//ncol, counter%ncol].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
    plt.setp(axes[counter//ncol, counter%ncol].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

    # turn grid on
    axes[counter//ncol, counter%ncol].grid(color=(.9, .9, .9)); axes[counter//ncol, counter%ncol].set_axisbelow(True)
    # axes[counter//ncol, counter%ncol].set_title(str(i) + '% off', fontname='Helvetica', fontsize=22)

    axes[counter//ncol, counter%ncol].set_yticks([20, 50, 100, 200, 500, 1000, 2000, 3000])
    axes[counter//ncol, counter%ncol].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    # add legend
    handles, labels = axes[counter//ncol, counter%ncol].get_legend_handles_labels()
    # print(handles, labels)
    axes[counter//ncol, counter%ncol].legend(handles, labels, prop={'size': 20})
    

plt.gcf().subplots_adjust(bottom=0.15, left = 0.035, right = 0.985)
plt.subplots_adjust(wspace=0.35, hspace=0.35)
# figure_name = './images/sp_buy_dips_since_' + str(min_year) + '.png'
figure_name = './images/8 - visualizing dip buys/dip_buy_visualization_' + str(min_year) + '_' + str(max_year) + '.png'
plt.savefig(figure_name, dpi = 250)
plt.show()




### CALCULATE VARIANCE IN RETURNS VS. DIP BUY STRATEGY 

dip = -50

buy_table_dip = pd.read_csv('./data/buy tables/buy_table_dip=' + str(dip) + '.csv')

sp_buys = buy_table_dip.drop(columns=buy_table_dip.columns[-3:]).drop(columns=['shares owned'])
sp_buys['date'] = pd.to_datetime(sp_buys['date'])

returns_interval = pd.DataFrame(columns=['time interval', 'start year', 'end year', 'start price', 'end price', 'return (%)', 'annual return (%)'])
returns_across_time_intervals = pd.DataFrame(columns=['time interval', 'avg return (%)', 'avg annual return (%)', 'annual return stdev (%)'])

total_time = int((sp_buys['date'].iloc[-1] - sp_buys['date'].iloc[0]).days / 365)

return_vs_dip_buy_interval = pd.DataFrame(columns=['dip', 'holding period', 'start year', 'end year', 'buy days', 'buy days (%)', 'return (%)'])

for i in range(2, total_time, 1):
    year_span = i - 1
    
    for j in range(1928, 2020 - year_span + 1, 1):
        
        # get start and end price over given time interval
        start_year = j
        end_year = j + year_span
        
        print(year_span, start_year, end_year)
        print('***')
        
        sp_buys_interval = sp_buys[(sp_buys['date'] >= pd.Timestamp(start_year, 1, 1, 12)) & (sp_buys['date'] <= pd.Timestamp(end_year, 1, 1, 12))]
        
        if len(sp_buys_interval) > 0:
        
            # add summary columns: cash balance, market value of shares, and current gain/loss
            sp_buys_interval['shares owned'] = sp_buys_interval['shares bought'].cumsum()
            sp_buys_interval['cash balance'] = sp_buys_interval['cash spend'].cumsum()
            sp_buys_interval['market value'] = sp_buys_interval['shares owned'] * sp_buys_interval['close']
            sp_buys_interval['current gain'] = sp_buys_interval['market value'] + sp_buys_interval['cash balance']
            
            # aggregate buy table to annual cash inflows/outflows for IRR calculation
            cash_outflows_yearly = sp_buys_interval.groupby(sp_buys_interval['date'].dt.year).sum()['cash spend']
            cash_inflow_final = pd.Series(sp_buys_interval.iloc[-1]['market value'], index=[sp_buys_interval.iloc[-1]['date'].year + 1])
        
            # last cash flow is implicit sale of entire position    
            cash_flows = cash_outflows_yearly.append(pd.Series(cash_inflow_final.values, index=[sp_buys_interval.iloc[-1]['date'].year + 1]))
            
            # calculate internal rate (IRR) of return of this cash flow schedule
            irr = np.irr(cash_flows)
        
            
            # build growing table of returns vs. threshold for dip buying
            return_vs_dip_buy_interval_temp = pd.DataFrame({'dip': dip,
                                                   'holding period': end_year - start_year,
                                                   'start year': start_year,
                                                   'end year': end_year,
                                                   'buy days': int(sp_buys_interval['shares owned'].iloc[-1]),
                                                   'buy days (%)': 100*np.round(sp_buys_interval['shares owned'].iloc[-1] / len(sp_buys_interval), 4),
                                                   'return (%)': 100*np.round(irr, 4)}, index=[end_year - start_year])
            
            return_vs_dip_buy_interval = return_vs_dip_buy_interval.append(return_vs_dip_buy_interval_temp)

return_vs_dip_buy_interval.to_csv('./data/returns_across_time_intervals_dip_buy_' + str(dip) + '_full.csv', index=False)






# create seaborn box + strip plot
dip = -10
return_vs_dip_buy_interval = pd.read_csv('./data/7 - returns across time intervals dip buy/returns_across_time_intervals_dip_buy_' + str(dip) + '_full.csv')

import seaborn as sns
fig, ax = plt.subplots(1, 1, figsize = (20, 20))

ax = sns.boxplot(x = 'holding period', y = 'return (%)', data = return_vs_dip_buy_interval, 
                 showfliers = False, order = list(set(return_vs_dip_buy_interval['holding period'])), linewidth = 1)
ax = sns.stripplot(x = 'holding period', y = 'return (%)', data = return_vs_dip_buy_interval,
                 order = list(set(return_vs_dip_buy_interval['holding period'])), jitter = 0.25, size = 5,
                 linewidth = 1, edgecolor = 'black', alpha = 0.5)

ax.axhline(y=0, linestyle=':', linewidth=3, color='grey')

# set axis properties
plt.xticks(fontname = 'Helvetica', fontsize = 42)
plt.yticks(fontname = 'Helvetica', fontsize = 42)

import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

plt.xlabel('Time horizon (years)', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Annual return (%)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(0, 20); ax.yaxis.labelpad = 25
# ax.set_ylim(-40, 50); ax.yaxis.labelpad = 25

ax.set_xlim(-1, total_time); ax.xaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3)

# turn grid on
plt.grid(color=(.75, .75, .75))
plt.grid(color=(.75, .75, .75))

figure_name = './images/9 - return vs. dip buy intervals/returns_vs_interval_dip_buy_' + str(dip) + '_1955_2020.png'

plt.tight_layout()
plt.savefig(figure_name, dpi = 150)
plt.show()




### CREATE SUMMARY OF RETURN VS DIP BUY OVER INTERVALS
return_vs_dip_buy_interval_0  = pd.read_csv('./data/7 - returns across time intervals dip buy/returns_across_time_intervals_dip_buy_0_full.csv')
return_vs_dip_buy_interval_10 = pd.read_csv('./data/7 - returns across time intervals dip buy/returns_across_time_intervals_dip_buy_-10_full.csv')
return_vs_dip_buy_interval_20 = pd.read_csv('./data/7 - returns across time intervals dip buy/returns_across_time_intervals_dip_buy_-20_full.csv')
return_vs_dip_buy_interval_30 = pd.read_csv('./data/7 - returns across time intervals dip buy/returns_across_time_intervals_dip_buy_-30_full.csv')

return_vs_dip_buy_all = pd.concat([return_vs_dip_buy_interval_0, return_vs_dip_buy_interval_10, return_vs_dip_buy_interval_20, return_vs_dip_buy_interval_30])


# plot comparison
fig, ax = plt.subplots(1, 1, figsize = (20, 20))

# Plot the responses for different events and regions
sns.lineplot(x = 'holding period', 
             y = 'return (%)', 
             data = return_vs_dip_buy_all, 
             hue='dip',
             palette=sns.color_palette("husl", 4),
             linewidth=5)

# set axis properties
plt.xticks(fontname = 'Helvetica', fontsize = 42)
plt.yticks(fontname = 'Helvetica', fontsize = 42)

import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

plt.xlabel('Time horizon (years)', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Annual return (%)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(0, 16); ax.yaxis.labelpad = 25

ax.set_xlim(0, total_time); ax.xaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3, color=(.5, .5, .5))

# turn grid on
sns.set_style("whitegrid")

# legend
l = plt.legend(borderpad=1, labelspacing = 1.5)
l.get_frame().set_linewidth(3)
l.get_texts()[0].set_text('Dip (%)')
l.get_texts()[4].set_text('   0')

for line in l.get_lines():
    line.set_linewidth(5)
    
for t in l.get_texts():
    t.set_ha('right')
    t.set_position((260,0))
    
plt.setp(ax.get_legend().get_texts(), fontsize='45', va='center') 
plt.setp(ax.get_legend().get_title(), fontsize='45') 

figure_name = './images/9 - return vs. dip buy intervals/returns_vs_interval_dip_buy_0_to_30_1955_2020.png'

# plt.tight_layout()
plt.savefig(figure_name, dpi = 150)
plt.show()





### COMPARE CONTINUOUS BUYING VS. ONE-TIME BUYS
one_time_buys = pd.read_csv('./data/5 - returns across time intervals/returns_across_time_intervals_full.csv')
one_time_buys_2 = one_time_buys[one_time_buys['start year'] >= 1955]
one_time_buys_3 = one_time_buys_2[['time interval', 'annual return (%)']]

continuous_buys = pd.read_csv('./data/7 - returns across time intervals dip buy/returns_across_time_intervals_dip_buy_0_full.csv')
continuous_buys_2 = continuous_buys[['holding period', 'return (%)']]


# plot comparison
fig, ax = plt.subplots(1, 1, figsize = (20, 20))

# Plot the responses for different events and regions
sns.lineplot(x = 'holding period', 
             y = 'return (%)', 
             data = continuous_buys_2, 
             color=sns.color_palette("husl", 4)[0],
             linewidth=5)

sns.lineplot(x = 'time interval', 
             y = 'annual return (%)', 
             data = one_time_buys_3, 
             color=sns.color_palette("husl", 4)[1],
             linewidth=5)

# set axis properties
plt.xticks(fontname = 'Helvetica', fontsize = 42)
plt.yticks(fontname = 'Helvetica', fontsize = 42)

import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

plt.xlabel('Time horizon (years)', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Annual return (%)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(0, 16); ax.yaxis.labelpad = 25

ax.set_xlim(0, total_time); ax.xaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3, color=(.5, .5, .5))

# turn grid on
sns.set_style("whitegrid")

# legend
l = plt.legend(borderpad=1, labelspacing = 1.5)
l.get_frame().set_linewidth(3)
# l.get_texts()[0].set_text('Dip (%)')
# l.get_texts()[4].set_text('   0')

for line in l.get_lines():
    line.set_linewidth(5)
    
for t in l.get_texts():
    t.set_ha('right')
    t.set_position((260,0))
    
plt.setp(ax.get_legend().get_texts(), fontsize='45', va='center') 
plt.setp(ax.get_legend().get_title(), fontsize='45') 

figure_name = './images/9 - return vs. dip buy intervals/returns_vs_interval_dip_buy_0_to_30_1955_2020.png'

# plt.tight_layout()
# plt.savefig(figure_name, dpi = 150)
plt.show()
