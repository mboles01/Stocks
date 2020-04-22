# # Historical S&P 500 analysis

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Stocks/') 

# load packages
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

### IMPORT AND COMBINE DATA SETS



# # S&P 500 daily closing value 1986 - 2018
# sp_old_raw = pd.read_csv('./data/sp500_to_29jun2018.csv')

# sp_new_raw_temp1 = pd.read_csv('./data/sp500_to_9apr2020.csv')
# sp_new_raw_temp2 = sp_new_raw_temp1[['Effective date ', 'S&P 500']]
# sp_new_raw = sp_new_raw_temp2[sp_new_raw_temp2['Effective date '].notna()]

# sp_old = pd.DataFrame({'date': pd.to_datetime(sp_old_raw['date']),
#                        'close': sp_old_raw['close']})

# sp_new = pd.DataFrame({'date': pd.to_datetime(sp_new_raw['Effective date ']),
#                        'close': sp_new_raw['S&P 500']})

# # combine old and new data sets
# sp = pd.concat([sp_old, sp_new]).drop_duplicates().reset_index(drop=True)




# # S&P 500 daily closing value 1927 - 2020
# sp_old_raw = pd.read_csv('./data/SP.csv')

# # # recast date as datetime
# # sp_old = pd.DataFrame({'date': pd.to_datetime(sp_old_raw['Date']),
# #                         'close': sp_old_raw['Close']})

# sp = sp_old_raw

# # save combined data set
# sp.to_csv('./data/sp500_all.csv', index=False)


# Load combined S&P 500 data set
sp = pd.read_csv('./data/sp500_all.csv')
sp['date'] = pd.to_datetime(sp['Date'])
sp['close'] = sp['Close']
sp = sp[['date', 'close']]
sp['log close'] = np.log(sp['close'])
sp[::1000]


### PLOT S&P 500 DATA

# linear scale
fig, axes = plt.subplots(1, 2, figsize = (18,7))
x = sp['date']
y = sp['close']
axes[0].plot(x, y, 'blue')
axes[0].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')

# log scale
axes[1].plot(x, y, 'blue')
axes[1].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_yscale('log')
axes[1].set_yticks([10, 20, 50, 100, 200, 500, 1000, 2000, 3000])
axes[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# set axis tick label properties
plt.setp(axes[0].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[0].get_yticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# turn grid on
axes[0].grid(color=(.9, .9, .9))
axes[1].grid(color=(.9, .9, .9))

figure_name = './images/sp_lin_log_2.png'

# plt.savefig(figure_name, dpi = 250)

plt.show()




# FIT PRICE DATA TO EXPONENTIAL

from scipy.optimize import curve_fit

import matplotlib.dates as mdates
x_num = mdates.date2num(sp['date']) - min(mdates.date2num(sp['date']))
x_date = [date.date() for date in mdates.num2date(x_num + min(mdates.date2num(sp['date'])))]
# x_span = np.linspace(x_num.min(), x_num.max(), len(x_num))



# # exponential fit: y = a * exp(bx) + c      [continuously compounding]
# def exp_function_3(x, a, b, c):
#     return a * np.exp(b * x) + c

# a1, a2 = [0.99, 1.01]
# b1, b2 = [-1, 1]
# c1, c2 = [10, 11]
# popt, pcov = curve_fit(exp_function_3, x_num, sp['close'], 
#                         absolute_sigma=False, maxfev=2000,
#                         bounds=([a1, b1, c1], [a2, b2, c2]))
# print(popt)

# sp_fit_temp = pd.DataFrame({'date': x_num, 
#                             'fit price': exp_function_3(x_num, popt[0], popt[1], popt[2])})

# sp_fit = pd.DataFrame({'date': x_date, 'fit price': sp_fit_temp['fit price']})



# # exponential fit: y = exp(ax) + b      [continuously compounding]
# def exp_function_2(x, a, b):
#     return np.exp(a * x) + b

# a1, a2 = [-1, 1]
# b1, b2 = [10, 11]
# popt, pcov = curve_fit(exp_function_2, x_num, sp['close'], 
#                         absolute_sigma=False, maxfev=2000,
#                         bounds=([a1, b1], [a2, b2]))
# print(popt)

# sp_fit_temp = pd.DataFrame({'date': x_num, 
#                             'fit price': exp_function_2(x_num, popt[0], popt[1])})

# sp_fit = pd.DataFrame({'date': x_date, 'fit price': sp_fit_temp['fit price']})


# annual compounding
def annual_compound(x, a, b):
    return (1 + a/365)**x + sp['close'].iloc[0] # + b

a1, a2 = [-100, 100]
# b1, b2 = [5, 15]
popt, pcov = curve_fit(annual_compound, x_num, sp['close'], 
                        absolute_sigma=False, maxfev=2000,
                        bounds=(a1, a2))
                        # bounds=([a1, b1], [a2, b2]))
print(popt)

sp_fit_temp = pd.DataFrame({'date': x_num, 
                            'fit price': annual_compound(x_num, popt[0], popt[1])})

sp_fit = pd.DataFrame({'date': x_date, 'fit price': sp_fit_temp['fit price']})



### PLOT S&P 500 DATA WITH EXPONENTIAL FIT

x_fit = sp_fit['date']
y_fit = sp_fit['fit price']

# linear scale
fig, axes = plt.subplots(1, 2, figsize = (18,7))
x = sp['date']
y = sp['close']
axes[0].plot(x, y, 'blue')
axes[0].plot(x_fit, y_fit, 'red')
axes[0].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')

# log scale
axes[1].plot(x, y, 'blue')
axes[1].plot(x_fit, y_fit, 'red')
axes[1].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_yscale('log')
axes[1].set_yticks([10, 20, 50, 100, 200, 500, 1000, 2000, 3000])
axes[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# set axis tick label properties
plt.setp(axes[0].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[0].get_yticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# turn grid on
axes[0].grid(color=(.9, .9, .9))
axes[1].grid(color=(.9, .9, .9))

# set up text box
props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)

textbox_1 = r'$P(t) = (1 + a) + b$'
textbox_2 = '$a$ = %0.6f \n$b$ = %0.3f' % (popt[0], popt[1]) #+ '\n$R^{2}$ = %5.2f' % r_squared_all + '\n$n$ = %d' % counts

# turn grid on
axes[0].grid(color=(.9, .9, .9))
axes[1].grid(color=(.9, .9, .9))

axes[0].text(0.05, 0.95, 
         textbox_1, 
         transform = axes[0].transAxes, 
         fontsize = 18, 
         fontname = 'Helvetica', 
         horizontalalignment = 'left',
         verticalalignment = 'top', 
         bbox = props_1)

axes[0].text(0.05, 0.85, 
         textbox_2, 
         transform = axes[0].transAxes, 
         fontsize = 18, 
         fontname = 'Helvetica', 
         horizontalalignment = 'left',
         verticalalignment = 'top', 
         bbox = props_2)

for tick in axes[0].get_xticklabels():
    tick.set_fontname('Helvetica')
for tick in axes[0].get_yticklabels():
    tick.set_fontname('Helvetica')

# title
fig.suptitle('S&P 500: exponential fit', fontsize = 22, fontname = 'Helvetica')
plt.subplots_adjust(wspace = 0.25)
figure_name = './images/sp_lin_log_fit.png'
plt.savefig(figure_name, dpi = 250)
plt.show()



# linear fit of ln-transformed data - continuous compounding
def linear_ln(x, a, b):
    # return a * x + b
    # return a * x + sp['log close'].iloc[0]
    return a * x/365 + np.log(b)

a1, a2 = [0.00001, 1]
b1, b2 = [1, 10]

popt, pcov = curve_fit(linear_ln, x_num, sp['log close'], 
                        absolute_sigma=False, maxfev=2000,
                        # bounds=(0.00001, 1))
                        bounds=([a1, b1], [a2, b2]))
print(popt)

sp_fit = pd.DataFrame({'date': x_date, 
                       # 'fit price': np.exp(linear_ln(x_num, popt[0]))})
                        'fit price': np.exp(linear_ln(x_num, popt[0], popt[1]))})



### PLOT S&P 500 DATA WITH EXPONENTIAL FIT

x_fit = sp_fit['date']
y_fit = sp_fit['fit price']

# linear scale
fig, axes = plt.subplots(1, 2, figsize = (12,5))
x = sp['date']
y = sp['close']
axes[0].plot(x, y, 'blue')
axes[0].plot(x_fit, y_fit, 'red')
axes[0].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[0].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')

# log scale
y = sp['close']
axes[1].plot(x, y, 'blue')
axes[1].plot(x_fit, y_fit, 'red')
axes[1].set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
axes[1].set_yscale('log')
axes[1].set_yticks([10, 20, 50, 100, 200, 500, 1000, 2000, 3000])
axes[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# set axis tick label properties
plt.setp(axes[0].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(axes[0].get_yticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(axes[1].get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# set up text box
props = dict(facecolor='white', edgecolor='none', alpha=0.67)

textbox_1 = r'$P(t) = exp(at) + b$'
# textbox_2 = '$a$ = %0.6f' % (popt[0]) #+ '\n$R^{2}$ = %5.2f' % r_squared_all + '\n$n$ = %d' % counts
textbox_2 = '$a$ = %0.4f \n$b$ = %0.3f' % (popt[0], popt[1]) #+ '\n$R^{2}$ = %5.2f' % r_squared_all + '\n$n$ = %d' % counts

# turn grid on
axes[0].grid(color=(.9, .9, .9))
axes[1].grid(color=(.9, .9, .9))

axes[0].text(0.05, 0.95, 
         textbox_1, 
         transform = axes[0].transAxes, 
         fontsize = 18, 
         fontname = 'Helvetica', 
         horizontalalignment = 'left',
         verticalalignment = 'top', 
         bbox = props)

axes[0].text(0.05, 0.85, 
         textbox_2, 
         transform = axes[0].transAxes, 
         fontsize = 18, 
         fontname = 'Helvetica', 
         horizontalalignment = 'left',
         verticalalignment = 'top', 
         bbox = props)

axes[1].text(0.05, 0.95, 
         textbox_1, 
         transform = axes[1].transAxes, 
         fontsize = 18, 
         fontname = 'Helvetica', 
         horizontalalignment = 'left',
         verticalalignment = 'top', 
         bbox = props)

axes[1].text(0.05, 0.85, 
         textbox_2, 
         transform = axes[1].transAxes, 
         fontsize = 18, 
         fontname = 'Helvetica', 
         horizontalalignment = 'left',
         verticalalignment = 'top', 
         bbox = props)

for tick in axes[0].get_xticklabels():
    tick.set_fontname('Helvetica')
for tick in axes[0].get_yticklabels():
    tick.set_fontname('Helvetica')

# title
fig.suptitle('S&P 500: exponential fit', fontsize = 22, fontname = 'Helvetica')
plt.subplots_adjust(wspace = 0.25)
figure_name = './images/sp_lin_log_fit.png'
plt.savefig(figure_name, dpi = 250)
plt.show()









# add daily change, high price info to dataframe
sp_diffs = pd.DataFrame({'date': sp['date'],
                         'close': sp['close'],
                         'daily change': round(sp.diff()['close'],4),
                         'daily change pct': round(100*sp.diff()['close']/sp['close'],3),
                         'high to date': sp['close'].cummax(),
                         'off from high': round(100*(sp['close'] - sp['close'].cummax()) / sp['close'].cummax(),3)})
sp_diffs[:5]


# # filter out older data
year = 1986
# newerthan = pd.Timestamp(year, 1, 1, 12)
# sp_diffs = sp_diffs[sp_diffs['date'] > newerthan]
# sp_diffs[:5]


### PLOT HISTOGRAM OF DAILY CHANGES ###

# create textbox
data = sp_diffs['daily change pct'][1:]
average = np.nanmean(data)
median = np.nanmedian(data)
stdev = np.std(data)
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$Daily$ $change$ (%%) \nAverage = %s \nMedian = %s \nStdev = %s' % (round(average,3), round(median,3), round(stdev,3))

from plotfunctions_1 import plot_hist
save=False
binwidth = 0.2
xmin = -3; xmax = 3
ymin = 0; ymax = 1200
xlabel = 'Daily change (%)'; ylabel = 'Counts (days)'
figure_name = './images/daily_changes_since_' + str(year) + '.png'
plot_hist(data, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, save, figure_name)

# # look at largest changes in either direction
# big_drops = sp_diffs.sort_values('daily change pct')[:25]
# big_gains = sp_diffs.sort_values('daily change pct', ascending=False)[:25]

# # get daily changes (%) n-tiles
# n = 5
# change_quantiles = pd.DataFrame({'daily change': pd.qcut(sp_diffs['daily change pct'][1:], n).value_counts().sort_index().reset_index()['index'],
#                              'counts': pd.qcut(sp_diffs['daily change pct'][1:], n).value_counts().sort_index().reset_index()['daily change pct']})
# change_quantiles


### PLOT HISTOGRAM OF PERCENT OFF HIGH ###

# linear scale

# create textbox
data = sp_diffs['off from high'][1:]
average = np.nanmean(data)
median = np.nanmedian(data)
stdev = np.std(data)
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$Down$ $from$ $high$ (%%) \nAverage = %s \nMedian = %s \nStdev = %s' % (round(average,3), round(median,3), round(stdev,3))

from plotfunctions_1 import plot_hist
save=False
binwidth = 1
xmin = -60; xmax = 0
ymin = 0; ymax = 1700
xlabel = 'Down from high (%)'; ylabel = 'Counts (days)'
figure_name = './images/down_from_high_since_' + str(year) + '.png'
plot_hist(data, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, save, figure_name)

# log scale

binwidth = 1
xmin = -60; xmax = 0
ymin = 0; ymax = 2500
yticks = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
xlabel = 'Down from high (%)'; ylabel = 'Counts (days)'

from plotfunctions_1 import plot_hist_log_y
save=False
binwidth = 1
xlabel = 'Down from high (%)'; ylabel = 'Counts (days)'
figure_name = './images/down_from_high_log_since_' + str(year) + '.png'
plot_hist_log_y(data, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, yticks, save, figure_name)

### PLOT JOINT PLOT OF PERCENT OFF HIGH ###


# joint plot: off from high vs. year
import seaborn as sns
import datetime as dt

# fig, ax = plt.subplots(1, 1, figsize = (7,7))
x = 2020 - (sp_diffs['date'].iloc[-1] - sp_diffs['date'][1:]).dt.days/365
y = sp_diffs['off from high'][1:]
g = sns.jointplot(x, y, kind="hex", color="blue")

# JointGrid has a convenience function
# g.set_axis_labels('x', 'y', rotation=0, fontsize=16, fontname = 'Helvetica')
# g.ax_joint.set_xticklabels(g.ax_joint.get_xmajorticklabels(), fontsize = 16)
# p.set_yticklabels(p.get_yticks(), size = 15)

# set axis tick label properties
# plt.setp(ax.get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
# plt.setp(ax.get_yticklabels(), fontsize=14, fontname = 'Helvetica')
plt.show()



# # get down range (% off) n-tiles with n = 10
# off_quantiles = pd.DataFrame({'down range': pd.qcut(sp_diffs['off from high'][1:], 10).value_counts().sort_index().reset_index()['index'],
#                              'counts': pd.qcut(sp_diffs['off from high'][1:], 10).value_counts().sort_index().reset_index()['off from high']})
# # off_quantiles.iloc[0][0] = pd.Interval(left = -56.776, right = -29.004, closed='right')
# off_quantiles




## Line graph: frequency of closing given amount below record high

# get fraction of closes below x% of record high
fract_pct_off = pd.DataFrame(columns=['percent off', 'fraction of days'])
for i in np.arange(0,58,0.1):
    pct_off = round(i,1)
    fract_pct_off_temp = pd.DataFrame({'percent off': pct_off, 
                                      'fraction of days': round(100*len(sp_diffs[sp_diffs['off from high'] <= -pct_off]) / len(sp_diffs), 3)}, index=[0])
    fract_pct_off = fract_pct_off.append(fract_pct_off_temp, ignore_index=True)
fract_pct_off = fract_pct_off[::-1].reset_index().iloc[:,1:]

fract_pct_off.to_csv('./data/fract_pct_off.csv', index=False)


### plot line of close fraction vs. off %

# linear scale
fig, axes = plt.subplots(1, 2, figsize = (18,7))
x = fract_pct_off['percent off']
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
plt.setp(axes[0].get_xticklabels(), rotation=0, fontsize=18, fontname = 'Helvetica')
plt.setp(axes[1].get_xticklabels(), rotation=0, fontsize=18, fontname = 'Helvetica')
plt.setp(axes[0].get_yticklabels(), fontsize=18, fontname = 'Helvetica')
plt.setp(axes[1].get_yticklabels(), fontsize=18, fontname = 'Helvetica')

# turn grid on
axes[0].grid(color=(.9, .9, .9))
axes[1].grid(color=(.9, .9, .9))

figure_name = './images/fraction_closes_down_since' + '.png' # + str(year)

# plt.savefig(figure_name, dpi = 250)
plt.subplots_adjust(wspace=0.25)
plt.show()



# print table showing fraction of closes below x% of record high

# fract_pct_off = pd.DataFrame(columns=['percent off', 'fraction of days (%)'])
# for i in np.arange(0,60,0.1):
#     pct_off = i
#     fract_pct_off_temp = pd.DataFrame({'percent off': -i, 
#                                       'fraction of days (%)': round(100*len(sp_diffs[sp_diffs['off from high'] <= -pct_off]) / len(sp_diffs), 2)}, index=[0])
    
#     fract_pct_off = fract_pct_off.append(fract_pct_off_temp, ignore_index=True)
    
fract_pct_off = fract_pct_off[::-1].reset_index().iloc[:,1:]
# fract_pct_off.style.hide_index()

sp_diffs.to_csv('./data/sp_diffs.csv', index=False)


