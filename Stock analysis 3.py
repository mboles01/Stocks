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
sp_changes = pd.read_csv('./data/sp_changes.csv')
# return_vs_dip_buy = pd.read_csv('./data/return_vs_dip_buy.csv')

# format date column to datetime
sp_changes['date'] = pd.to_datetime(sp_changes['date'])

# select date range to filter
min_year = 1955
max_year = 2020
sp_3 = sp_changes[(sp_changes['date'] >= pd.Timestamp(min_year, 1, 1, 12)) & 
                  (sp_changes['date'] <= pd.Timestamp(max_year+1, 1, 1, 12))]
sp_3


# are today's and tomorrow's movements correlated?

sp_4 = pd.DataFrame({'date': sp_3['date'],
                     'change day 1': sp_3['daily change pct'],
                     'change day 2': sp_3['daily change pct'].shift(-1)})

change_corr = pd.DataFrame(columns=['min change day 1', 'corr with change day 2'])
for i in np.arange(0,10.1,0.1):
    # sp_selected = sp_4[sp_4['change day 1'] > round(i,1)]
    sp_selected = sp_4[(sp_4['change day 1'] > round(i,1)) | (sp_4['change day 1'] < -round(i,1))]
    change_corr_temp = pd.DataFrame({'min change day 1': round(i,1),
                                    'corr with change day 2': round(sp_selected.corr().iloc[0][1],3)}
                                    , index=[0])
    change_corr = change_corr.append(change_corr_temp)



### PLOT CORRELATION IN MOVEMENT ACROSS CONSECUTIVE DAYS ###

# line plot: correlation vs. minimum change

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
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
ax.set_xlim(-0.1, 10); ax.xaxis.labelpad = 15
ax.set_ylim(-1, 0.1); ax.yaxis.labelpad = 15

# turn grid on
ax.grid(color=(.9, .9, .9))

# plt.subplots_adjust(wspace = 0.35)
plt.gcf().subplots_adjust(bottom=0.15, left=0.2)

figure_name = './images/movement_correlation_across_days_since_' + str(min_year) + '.png' 
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








