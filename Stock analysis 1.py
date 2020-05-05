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


# Load combined S&P 500 data set
sp = pd.read_csv('./data/raw/sp500_all.csv')   # https://finance.yahoo.com/quote/%5EGSPC/history/ 
sp['date'] = pd.to_datetime(sp['Date'])
sp['close'] = sp['Close']
sp = sp[['date', 'close']]
sp['ln close'] = np.log(sp['close'])
sp[::1000]


# select date range to filter
min_year = 1955
max_year = 2020
sp = sp[(sp['date'] >= pd.Timestamp(min_year, 1, 1, 12)) & 
            (sp['date'] <= pd.Timestamp(max_year+1, 1, 1, 12))]
sp


### PLOT S&P 500 DATA

# linear scale
fig, axes = plt.subplots(1, 2, figsize = (12,5))
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

figure_name = './images/1 - price history/sp_' + str(min_year) + '_' + str(max_year) + '.png'
plt.subplots_adjust(wspace = 0.35)
plt.gcf().subplots_adjust(bottom=0.15)
# plt.savefig(figure_name, dpi = 250)
plt.show()




# FIT PRICE DATA

from scipy.optimize import curve_fit

import matplotlib.dates as mdates
x_num = mdates.date2num(sp['date']) - min(mdates.date2num(sp['date']))
x_date = [date.date() for date in mdates.num2date(x_num + min(mdates.date2num(sp['date'])))]
# x_span = np.linspace(x_num.min(), x_num.max(), len(x_num))




# annual compounding - fit prices on linear scale
def annual_compound(x, r, b):
    return (1 + r*365)**(x/365) + b #+ sp['close'].iloc[0] # + b

r1, r2 = [-100, 100]
# b1, b2 = [5, 125]     # Po for 1928
b1, b2 = [30, 400]      # Po for 1955

popt, pcov = curve_fit(annual_compound, x_num, sp['close'], 
                        absolute_sigma=False, maxfev=2000,
                        # bounds=(r1, r2))
                        bounds=([r1, b1], [r2, b2]))
print(popt)

sp_fit = pd.DataFrame({'date': x_date, 'fit price': annual_compound(x_num, popt[0], popt[1])})

# calculate fit quality R2
ss_residual = np.sum((sp['close'] - sp_fit['fit price'])**2)   # residual sum of squares
ss_total = np.sum((sp['close'] - np.mean(sp_fit['fit price']))**2)   # total sum of squares
r_squared_all = 1 - (ss_residual / ss_total)



### PLOT S&P 500 DATA WITH FIT

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
props = dict(facecolor='white', edgecolor='none', alpha=0.67)

textbox_1 = r'$P(t)$ = (1 + $r)^t + P_o$'
textbox_2 = '$r$ = %0.2f %% \n$P_o$ = %0.2f' % (popt[0]*365*100, popt[1]) + '\n$R^{2}$ = %5.2f' % r_squared_all

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
# fig.suptitle('S&P 500 fit', fontsize = 22, fontname = 'Helvetica')
plt.subplots_adjust(wspace = 0.35)
plt.gcf().subplots_adjust(bottom=0.15)
figure_name = './images/1 - price history/sp_fit_linear_' + str(min_year) + '_' + str(max_year) + '.png'
# plt.savefig(figure_name, dpi = 250)
plt.show()





# annual compounding - fit prices on log scale

def linear_ln(x, r, b):
    return np.log((1 + r*365)**(x/365) + b)   # log = natural log (ln)

r1, r2 = [0.00001, 1]
b1, b2 = [5, 15]
b1, b2 = [10, 100]

popt, pcov = curve_fit(linear_ln, x_num, sp['ln close'], 
                        absolute_sigma=False, maxfev=2000,
                        bounds=([r1, b1], [r2, b2]))
print(popt)

sp_fit = pd.DataFrame({'date': x_date, 
                        'ln fit price': linear_ln(x_num, popt[0], popt[1]),
                        'fit price': np.exp(linear_ln(x_num, popt[0], popt[1]))})

# calculate fit quality R2
ss_residual = np.sum((sp['ln close'] - sp_fit['ln fit price'])**2)   # residual sum of squares
ss_total = np.sum((sp['ln close'] - np.mean(sp_fit['ln fit price']))**2)   # total sum of squares
r_squared_all = 1 - (ss_residual / ss_total)




### PLOT S&P 500 DATA WITH FIT

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

textbox_1 = r'ln( $P(t)$ ) = ln( $(1 + r)^t + P_o )$'
textbox_2 = '$r$ = %0.2f %% \n$P_o$ = %0.2f' % (popt[0]*365*100, popt[1]) + '\n$R^{2}$ = %5.2f' % r_squared_all

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
# fig.suptitle('S&P 500: exponential fit', fontsize = 22, fontname = 'Helvetica')
plt.subplots_adjust(wspace = 0.35)
plt.gcf().subplots_adjust(bottom=0.15)
figure_name = './images/1 - price history/sp_fit_log_' + str(min_year) + '_' + str(max_year) + '.png'
plt.savefig(figure_name, dpi = 250)
plt.show()
