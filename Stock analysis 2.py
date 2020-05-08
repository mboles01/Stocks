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

# sp_2.to_csv('./data/sp_changes.csv', index=False)






### PLOT HISTOGRAM OF DAILY CHANGES ###


# select date range to filter
min_year = 1928
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

fract_pct_off = pd.read_csv('./data/2 - fraction pct off/fract_pct_off_1955_2020.csv')
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





# # GET CORRELATIONS IN DAILY MOVEMENTS ACROSS CONSECUTIVE DAYS

# sp_4 = pd.DataFrame({'date': sp_3['date'],
#                       'change day 1': sp_3['daily change pct'],
#                       'change day 2': sp_3['daily change pct'].shift(-1)})

# change_corr = pd.DataFrame(columns=['min change day 1', 'corr with change day 2'])
# for i in np.arange(0,15.1,0.1):
#     # sp_selected = sp_4[sp_4['change day 1'] > round(i,1)]
#     sp_selected = sp_4[(sp_4['change day 1'] > round(i,1)) | (sp_4['change day 1'] < -round(i,1))]
#     change_corr_temp = pd.DataFrame({'min change day 1': round(i,1),
#                                     'corr with change day 2': round(sp_selected.corr().iloc[0][1],3)}
#                                     , index=[0])
#     change_corr = change_corr.append(change_corr_temp)

# change_corr.to_csv('./data/change_correlations_' + str(min_year) + '_' + str(max_year) + '.csv')



# line plot: correlation vs. minimum change

# change_corr = pd.read_csv('./data/change_correlations_1928_2020.csv')
change_corr = pd.read_csv('./data/change_correlations_1955_2020.csv')

fig, ax = plt.subplots(1, 1, figsize = (7, 7))
x = change_corr['min change day 1']
y = change_corr['corr with change day 2']
ax.plot(x, y, 'blue', linewidth=3, zorder=20)
ax.axhline(y=0, linestyle=':', linewidth=2, color='grey', zorder=10)

ax.set_xlabel('Minimum day 1 change (%)', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
ax.set_ylabel('Correlation with day 2 change', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')

# set axis tick label properties
plt.setp(ax.get_xticklabels(), fontsize=14, fontname = 'Helvetica')
plt.setp(ax.get_yticklabels(), fontsize=14, fontname = 'Arial')

# axis limits
ax.set_xlim(-0.1, 11); ax.xaxis.labelpad = 15
ax.set_ylim(-1, 0.05); ax.yaxis.labelpad = 15

# turn grid on
ax.grid(color=(.9, .9, .9))

# plt.subplots_adjust(wspace = 0.35)
plt.gcf().subplots_adjust(bottom=0.15, left=0.2)

figure_name = './images/5 - movement correlations/movement_correlation_across_days_' + str(min_year) + '_' + str(max_year) + '.png' 
plt.savefig(figure_name, dpi = 250)
plt.show()


# # scatter plot: day 1 change vs. day 2 change

# x = sp_4['change day 1']
# y = sp_4['change day 2']

# x_avg = np.mean(x)
# y_avg = np.mean(y)

# # scatter plot
# fig, ax = plt.subplots(1, 1, figsize=(7,7))
# ax.scatter(x,y, edgecolor='black', facecolor='blue', alpha=0.04)
# plt.xlim(-3, 3); plt.ylim(-3, 3) 
# plt.xlabel('Daily change (%)', fontsize = 18, fontname = 'Helvetica')
# plt.ylabel("Next day's change (%)", fontsize = 18, fontname = 'Helvetica')

# ax.tick_params(axis = 'x', labelsize = 14)
# ax.tick_params(axis = 'y', labelsize = 14)

# for tick in ax.get_xticklabels():
#     tick.set_fontname('Helvetica')
# for tick in ax.get_yticklabels():
#     tick.set_fontname('Helvetica')

# plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
# plt.show()



### CALCULATE TIME INTERVAL BETWEEN NEIGHBORING HIGHS ###

# sp_5 = sp_3[sp_3['off from high'] == 0]
# high_fraction = len(sp_5)/len(sp_3)

# neighboring_highs = pd.DataFrame({'high date': sp_5['date'],
#                                   'high close': sp_5['close'],
#                                   'next high date': sp_5['date'].shift(-1),
#                                   'next high close': sp_5['close'].shift(-1),
#                                   'date diff': (sp_5['date'].shift(-1) - sp_5['date']).astype('timedelta64[D]'),
#                                   'close diff': round(sp_5['close'].shift(-1) - sp_5['close'], 3),
#                                   'close diff (%)': round(100*(sp_5['close'].shift(-1) - sp_5['close'])/sp_5['close'], 3)})

# neighboring_highs.to_csv('./data/4 - time between neighboring highs/neighboring_highs.csv', index=False)

# plot histogram

neighboring_highs = pd.read_csv('./data/4 - time between neighboring highs/neighboring_highs.csv')

# create textbox
data = neighboring_highs['date diff']
# data = neighboring_highs['close diff (%)']
average = np.nanmean(data)
median = np.nanmedian(data)
stdev = np.std(data)
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$Time$ $between$ $highs$ $(days)$ \nMean = %0.1f \nMedian = %0.0f \nSt. dev. = %0.0f' % (average, median, stdev)
text_pos = 0.35

from plotfunctions_1 import plot_hist
save=False
binwidth = 10
xmin = -20; xmax = 300
ymin = 0; ymax = 1500
xlabel = 'Time between highs (days)'; ylabel = 'Counts'
figure_name = './images/6 - time between highs/time_between_highs_' + str(min_year) + '_' + str(max_year) + '.png'
plot_hist(data, binwidth, textbox, props, text_pos, xmin, xmax, ymin, ymax, xlabel, ylabel, save, figure_name)

from plotfunctions_1 import plot_hist_log_y
save=True
binwidth = 10
xmin = 0; xmax = 3000
ymin = 0; ymax = 1500; yticks = [1, 10, 100, 1000]
xlabel = 'Time between highs (days)'; ylabel = 'Counts'
figure_name = './images/6 - time between highs/time_between_highs_log_' + str(min_year) + '_' + str(max_year) + '.png'
plot_hist_log_y(data, binwidth, textbox, props, text_pos, xmin, xmax, ymin, ymax, xlabel, ylabel, yticks, save, figure_name)




# # how many new highs between "recessions"? - periods of a given time interval with no new records set

# last_highs = pd.DataFrame(columns=['high date', 'high close', 'next high date', 'next high close', 'date diff', 'close diff', 'close diff (%)', 'last high', 'remaining highs', 'min days no high'])
# remaining_highs = pd.DataFrame(columns=['high date', 'high close', 'next high date', 'next high close', 'date diff', 'close diff', 'close diff (%)', 'last high', 'remaining highs'])

# for i in np.arange(10, 1000, 10):
#     min_days_no_high = i
#     print(min_days_no_high)
#     remaining_highs_temp = neighboring_highs.copy()
    
#     remaining_highs_temp['last high'] = remaining_highs_temp['date diff'].apply(lambda x: 1 if x >= min_days_no_high else 0)
#     remaining_highs_temp['remaining highs'] = remaining_highs_temp.groupby(remaining_highs_temp['last high'].gt(0).cumsum()).cumcount(ascending=False)
    
#     last_highs_temp = remaining_highs_temp[remaining_highs_temp['last high'] == 1]
#     last_highs_temp['min days no high'] = int(min_days_no_high)
    
#     print(last_highs_temp)
    
#     remaining_highs = remaining_highs.append(remaining_highs_temp)
#     last_highs = last_highs.append(last_highs_temp)

# last_highs.to_csv('./data/4 - time between neighboring highs/last_highs.csv', index=False)
# remaining_highs.to_csv('./data/4 - time between neighboring highs/remaining_highs.csv', index=False)
 


### PLOT NUMBER OF NEW HIGHS SET VS. NUMBER OF DAYS WITHOUT A HIGH ###

last_highs = pd.read_csv('./data/4 - time between neighboring highs/last_highs.csv')

min_days = 100
max_days = 10000
min_days_no_high = 360
last_highs_filtered = last_highs[(last_highs['date diff'] >= min_days) & 
                                 (last_highs['date diff'] <= max_days) & 
                                 (last_highs['min days no high'] == min_days_no_high)]

# last_highs_filtered = last_highs[last_highs['min days no high'] == min_days_no_high]

# plot histogram
data = last_highs_filtered['remaining highs']

# create textbox
average = np.nanmean(data)
median = np.nanmedian(data)
stdev = np.std(data)
from scipy.stats import skew, kurtosis
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$New$ $highs$ \nMean = %0.1f \nMedian = %0.0f \nSt. dev. = %0.1f' % (average, median, stdev)
text_pos = 0.05

from plotfunctions_1 import plot_hist
save=False
binwidth = 10
xmin = -1; xmax = 130
ymin = 0; ymax = 3
yint = True
# xlabel = 'Number of new highs after first high in ' + str(min_days_no_high) + ' days'; ylabel = 'Counts'
xlabel = 'Number of new highs after first high in at least one year'; ylabel = 'Counts'
figure_name = './images/4 - down from high/number_of_highs.png'
plot_hist(data, binwidth, textbox, props, text_pos, xmin, xmax, ymin, ymax, yint, xlabel, ylabel, save, figure_name)



### PLOT S&P 500 DATA WITH LAST HIGHS ON TOP OF IT

fig, ax = plt.subplots(1, 1, figsize = (7,7))
x = sp['date']
y = sp['close']

last_highs_vline = pd.to_datetime(last_highs_filtered['high date'])

# log scale
import matplotlib
ax.plot(x, y, 'blue')
ax.set_xlabel('Year', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
ax.set_ylabel('Close', fontsize = 18, fontname = 'Helvetica', fontweight = 'bold')
ax.set_yscale('log')
ax.set_yticks([10, 20, 50, 100, 200, 500, 1000, 2000, 3000])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

for high in last_highs_vline:
    ax.axvline(high, linewidth=1, color='r', zorder=1)

# set axis tick label properties
plt.setp(ax.get_xticklabels(), rotation=30, fontsize=14, fontname = 'Helvetica')
plt.setp(ax.get_yticklabels(), fontsize=14, fontname = 'Helvetica')

# turn grid on
ax.grid(color=(.9, .9, .9))

figure_name = './images/4 - down from high/highs_taking_minimum_1y_to_retest.png'
plt.subplots_adjust(wspace = 0.35)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(figure_name, dpi = 250)
plt.show()










# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# # plot scatter data and line of best fit
# x = last_highs_filtered['date diff']
# y = last_highs_filtered['remaining highs']
# fig, ax = plt.subplots(1, 1, figsize=(7,7))
# ax.scatter(x, y, edgecolor='black', facecolor='blue', alpha=0.5)
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), '#ff4c00', linewidth=3)

# plt.xlim(0,1000)
# # plot fit
# x_axis_smooth = np.arange(min(year_age_median_price['Age']), max(year_age_median_price['Age'])+1, .1)
# plt.plot(x_axis_smooth, exp_function(x_axis_smooth, *popt), '#ff4c00', linewidth=3)

# set x- and y- labels
xlabel = 'Time since previous high (days)'
ylabel = 'Number of highs' 
# yscale = 1000

plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
plt.ylabel(ylabel, fontsize = 18, fontname = 'Helvetica')
ax.tick_params(axis = 'x', labelsize = 14)
ax.tick_params(axis = 'y', labelsize = 14)

# force integers on x-axis
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
# # set up text box
# props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
# props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)

# textbox_1 = r'$P(t) = a{\bullet}exp(-bt)$'
# textbox_2 = '$a$ = %5.0f \n$b$ = %0.3f' % (popt[0], popt[1]) + '\n$R^{2}$ = %5.2f' % r_squared_all
# textbox_3 = '$Half$ $life:$ ' + str(round(0.6931/popt[1], 2)) + ' $y$'

# ax.text(0.54, 0.95, textbox_1, transform = ax.transAxes, fontsize = 18, 
#         fontname = 'Helvetica', verticalalignment = 'top', bbox = props_1)

# ax.text(0.72, 0.85, textbox_2, transform = ax.transAxes, fontsize = 18, 
#         fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)

# ax.text(0.625, 0.65, textbox_3, transform = ax.transAxes, fontsize = 18, 
#         fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)


for tick in ax.get_xticklabels():
    tick.set_fontname('Helvetica')
for tick in ax.get_yticklabels():
    tick.set_fontname('Helvetica')

plt.rcParams['axes.unicode_minus'] = False
plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)

ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/yscale))
ax.yaxis.set_major_formatter(ticks)

figure_name = '../images/depr_by_model/' + str(counter) + '_' + str(model) + '.png'
plt.savefig(figure_name, dpi = 600)
plt.show()






























