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

# import data
# fract_pct_off = pd.read_csv('./data/fract_pct_off_1928_2020.csv')
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


### BUILD BUY TABLE

return_vs_dip_buy = pd.DataFrame(columns=['dip', 'buy days', 'buy days (%)', 'return (%)'])

sp_buys = sp_3

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




### PLOT LINE OF RETURN VS. DIP BUY

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
ax.grid(color=(.9, .9, .9)); 

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



# create visual display of buying dips for various cases

# get buy dates
start = 10
stop = 45
step = 5
sp_with_buy_dates = sp_changes.copy()
sp_with_buy_dates['buy'] = sp_changes['off from high'] < -n

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


    