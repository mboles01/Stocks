# Historical S&P 500 analysis

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Stocks/') 

# load packages
# import warnings; warnings.simplefilter('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





# Load combined S&P 500 data set
sp = pd.read_csv('./data/raw/sp500_all.csv')
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

# sp_2.to_csv('./data/sp_changes.csv', index=False)






### PLOT HISTOGRAM OF DAILY CHANGES ###


# select date range to filter
min_year = 1955
max_year = 2020
sp_3 = sp_2[(sp_2['date'] >= pd.Timestamp(min_year, 1, 1, 12)) & 
            (sp_2['date'] <= pd.Timestamp(max_year+1, 1, 1, 12))]
sp_3

# create textbox
data = sp_3['daily change pct'][1:]
average = np.nanmean(data)
median = np.nanmedian(data)
stdev = np.std(data)
from scipy.stats import skew, kurtosis
skew = skew(data) # negative skew - longer left-side tail
kurtosis = kurtosis(data) # positive kurtosis - more probability density at tails than normal dist
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$Daily$ $change$ (%%) \nMean = %0.3f \nMedian = %0.3f \nSt. dev. = %0.2f \nSkew = %0.2f \nKurtosis = %0.1f' % (average, median, stdev, skew, kurtosis)
text_pos = 0.05

from plotfunctions_1 import plot_hist
save=True
binwidth = 0.2
xmin = -4; xmax = 4
# ymin = 0; ymax = 3000  # from 1928
ymin = 0; ymax = 2500  # from 1955
xlabel = 'Daily change (%)'; ylabel = 'Counts (days)'
figure_name = './images/2 - daily changes/daily_changes_' + str(min_year) + '_' + str(max_year) + '.png'
plot_hist(data, binwidth, textbox, props, text_pos, xmin, xmax, ymin, ymax, xlabel, ylabel, save, figure_name)








    


# ### CALCULATE ANNUALIZED RETURNS ACROSS ALL POSSIBLE TIME INTERVALS ###

# returns_interval = pd.DataFrame(columns=['time interval', 'start year', 'end year', 'start price', 'end price', 'return (%)', 'annual return (%)'])
# returns_across_time_intervals = pd.DataFrame(columns=['time interval', 'avg return (%)', 'avg annual return (%)', 'annual return stdev (%)'])

# total_time = int((sp_3['date'].iloc[-1] - sp_3['date'].iloc[0]).days / 365)

# for i in range(2, total_time, 1):
#     year_span = i - 1
    
#     for j in range(1928, 2020 - year_span + 1, 1):
        
#         # get start and end price over given time interval
#         start_year = j
#         end_year = j + year_span
        
#         print(year_span, start_year, end_year)
#         print('***')
        
#         start_price = sp_2[(sp_2['date'] >= pd.Timestamp(start_year, 1, 1, 12))].iloc[0]['close']
#         end_price = sp_2[(sp_2['date'] >= pd.Timestamp(end_year, 1, 1, 12))].iloc[0]['close']
        
#         # append return data to growing dataframe 
#         returns_interval_temp = pd.DataFrame({'time interval': year_span,
#                                               'start year': start_year,
#                                               'end year': end_year, 
#                                               'start price': start_price,
#                                               'end price': end_price,
#                                               'return (%)': 100*((end_price / start_price) - 1),
#                                               'annual return (%)': 100*((end_price / start_price) ** (1/year_span) - 1)
#                                               }, index=[0])
#         returns_interval = returns_interval.append(returns_interval_temp, ignore_index = True)

#     # append statistical data to growing dataframe
#     returns_across_time_intervals_temp = pd.DataFrame({'time interval': year_span,
#                                                        'avg return (%)': round(np.nanmean(returns_interval['return (%)']),3),
#                                                        'avg annual return (%)': round(np.nanmean(returns_interval['annual return (%)']), 3),
#                                                        'annual return stdev (%)': round(np.std(returns_interval['annual return (%)']), 3),
#                                                        }, index=[0])

#     returns_across_time_intervals = returns_across_time_intervals.append(returns_across_time_intervals_temp, ignore_index = True)
        
    
# returns_across_time_intervals.to_csv('./data/returns_across_time_intervals_since_' + str(min_year) + '.csv', index=False)
# returns_interval.to_csv('./data/returns_across_time_intervals_full_since_' + str(min_year) + '.csv', index=False)









### PLOT HISTORICAL RETURNS BY INVESTMENT HORIZON ###

returns_interval = pd.read_csv('./data/returns_across_time_intervals_full.csv')

# # select date range to filter
min_year = 1928
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
                 linewidth = 1, edgecolor = 'black', alpha = 0.5)

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


### PLOT HISTOGRAM OF PERCENT OFF HIGH ###

# select date range to filter
min_year = 1955
max_year = 2020
sp_3 = sp_2[(sp_2['date'] >= pd.Timestamp(min_year, 1, 1, 12)) & 
            (sp_2['date'] <= pd.Timestamp(max_year+1, 1, 1, 12))]
sp_3

# create textbox
data = sp_3['off from high'][1:]
average = np.nanmean(data)
median = np.nanmedian(data)
stdev = np.std(data)
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$Down$ $from$ $high$ (%%) \nMean = %0.1f \nMedian = %0.1f \nSt. dev. = %0.1f' % (round(average,3), round(median,3), round(stdev,3))
text_pos = 0.05

# linear scale

from plotfunctions_1 import plot_hist
save=False
binwidth = 1
# xmin = -90; xmax = 0   # 1928
xmin = -55; xmax = 0   # 1955
ymin = 0; ymax = 2000
xlabel = 'Down from high (%)'; ylabel = 'Counts (days)'
figure_name = './images/4 - down from high/down_from_high_' + str(min_year) + '_' + str(max_year) + '.png'
plot_hist(data, binwidth, textbox, props, text_pos, xmin, xmax, ymin, ymax, xlabel, ylabel, save, figure_name)


# log scale

binwidth = 1
xmin = -90; xmax = 0   # 1928
# xmin = -55; xmax = 0   # 1955
ymin = 0; ymax = 2500
yticks = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
xlabel = 'Down from high (%)'; ylabel = 'Counts (days)'

from plotfunctions_1 import plot_hist_log_y
save=False
binwidth = 1
xlabel = 'Down from high (%)'; ylabel = 'Counts (days)'
figure_name = './images/4 - down from high/down_from_high_log_' + str(min_year) + '_' + str(max_year) + '.png'
plot_hist_log_y(data, binwidth, textbox, props, text_pos, xmin, xmax, ymin, ymax, xlabel, ylabel, yticks, save, figure_name)






### PLOT PERCENT OFF HIGH VS TIME ###

fig, ax = plt.subplots(1, 1, figsize = (7, 7))
x = 2020 - (sp_3['date'].iloc[-1] - sp_3['date'][1:]).dt.days/365
y = sp_3['off from high'][1:]
ax.plot(x, y, 'blue', zorder=20)
ax.axhline(y=0, linestyle=':', linewidth=2, color='grey', zorder=10)

ax.set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
ax.set_ylabel('Down from high (%)', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')

# set axis tick label properties
plt.setp(ax.get_xticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(ax.get_yticklabels(), fontsize=14, fontname = 'Arial')

# turn grid on
ax.grid(color=(.9, .9, .9))

figure_name = './images/4 - down from high/down_from_high_vs_time_' + str(min_year) + '_' + str(max_year) + '.png' 
plt.subplots_adjust(wspace = 0.35)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(figure_name, dpi = 250)
plt.show()







# ## Line graph: frequency of closing given amount below record high

# # select date range to filter
# min_year = 1955
# max_year = 2020
# sp_3 = sp_2[(sp_2['date'] >= pd.Timestamp(min_year, 1, 1, 12)) & 
#             (sp_2['date'] <= pd.Timestamp(max_year+1, 1, 1, 12))]
# sp_3

# max_pct_off = 58  # 1955
# # max_pct_off = 87  # 1928

# # get fraction of closes below x% of record high
# fract_pct_off = pd.DataFrame(columns=['percent off', 'fraction of days'])
# for i in np.arange(0,max_pct_off,0.1):
#     pct_off = round(i,1)
#     fract_pct_off_temp = pd.DataFrame({'percent off': pct_off, 
#                                       'fraction of days': round(100*len(sp_3[sp_3['off from high'] <= -pct_off]) / len(sp_3), 3)}, index=[0])
#     fract_pct_off = fract_pct_off.append(fract_pct_off_temp, ignore_index=True)
# fract_pct_off = fract_pct_off[::-1].reset_index().iloc[:,1:]

# fract_pct_off.to_csv('./data/fract_pct_off_' + str(min_year) + '_' + str(max_year) + '.csv', index=False)



### plot line of close fraction vs. off %

fract_pct_off = pd.read_csv('./data/fract_pct_off_1928_2020.csv')
# fract_pct_off = pd.read_csv('./data/fract_pct_off_1955_2020.csv')

# linear scale
fig, axes = plt.subplots(1, 2, figsize = (14,6))
x = -fract_pct_off['percent off']
y = fract_pct_off['fraction of days']
axes[0].plot(x, y, 'blue', linewidth=3)
axes[0].set_xlabel('Down from high (%)', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].set_ylabel('Fraction of closes (%)', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].set_ylim(-1,105)

# log scale
axes[1].plot(x, y, 'blue', linewidth=3)
axes[1].set_xlabel('Down from high (%)', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_ylabel('Fraction of closes (%)', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_yscale('log')
axes[1].set_ylim(0.001,175)
axes[1].set_yticklabels(['', '0.001', '0.01', '0.1', '1', '10', '100'])

# set axis tick label properties
plt.setp(axes[0].get_xticklabels(), rotation=0, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_xticklabels(), rotation=0, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[0].get_yticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# turn grid on
axes[0].grid(color=(.9, .9, .9))
axes[1].grid(color=(.9, .9, .9))

figure_name = './images/4 - down from high/fraction_closes_down_' + str(min_year) + '_' + str(max_year) + '.png'
plt.subplots_adjust(wspace = 0.35)
plt.gcf().subplots_adjust(bottom=0.15)

# plt.savefig(figure_name, dpi = 250)
plt.subplots_adjust(wspace=0.25)
plt.show()




# print table showing fraction of closes below x% of record high
# fract_pct_off = pd.DataFrame(columns=['percent off', 'fraction of days (%)'])
# for i in np.arange(0,60,0.1):
#     pct_off = i
#     fract_pct_off_temp = pd.DataFrame({'percent off': -i, 
#                                       'fraction of days (%)': round(100*len(sp_3[sp_3['off from high'] <= -pct_off]) / len(sp_3), 2)}, index=[0])
    
#     fract_pct_off = fract_pct_off.append(fract_pct_off_temp, ignore_index=True)
    
# fract_pct_off = fract_pct_off[::-1].reset_index().iloc[:,1:]
# fract_pct_off.style.hide_index()



