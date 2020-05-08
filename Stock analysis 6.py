#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:36:33 2020

@author: michaelboles
"""


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




# explore relationship between return rate and principle after n years

t = np.arange(0, 31, 2)
Po = 1
rates = np.arange(1, 16, 1)

growth = pd.DataFrame(index=t, columns=rates, dtype='float')
for time in t:
    print(time)
    for rate in rates:
        growth.at[time,rate] = round(Po * (1 + rate/100) ** time, 3)
        
growth_log = np.log(growth).transpose()
labels = growth.transpose()

# plot result

import seaborn as sns
fig, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(260, 0, n=9, as_cmap=True)
ax = sns.heatmap(growth_log, 
                 cmap=cmap, 
                 annot=labels,
                 annot_kws={"fontsize":18},
                 vmin=growth_log.min().min(), 
                 vmax=growth_log.max().max(),
                 square=True, 
                 linewidths=.5, 
                 cbar=False)
plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=24, fontname='Helvetica')
plt.setp(ax.get_yticklabels(), rotation=0, ha='right', fontsize=24, fontname='Helvetica')
plt.xlabel('\nTime horizon (y)', fontsize=28, fontname='Helvetica')
plt.ylabel('Return (%)\n', fontsize=28, fontname='Helvetica')
ax.invert_yaxis()
ax.set_facecolor([1,1,1])
ax.set_title('$Investment$ $return$: $single$ $deposit$\n', fontsize=32, fontname='Helvetica')

figure_name = './images/10 - hypothetical returns/hypothetical returns.png'
plt.savefig(figure_name, dpi = 250)
plt.show()




# same, but with recurring deposits - future value of annuity due (t.ly/7V0z)

t = np.arange(0, 31, 2)
Po = 1
rates = np.arange(1, 16, 1)

growth_recur_dep = pd.DataFrame(index=t, columns=rates, dtype='float')
for time in t:
    print(time)
    for rate in rates:
        growth_recur_dep.at[time,rate] = round(Po * (((1 + rate/100) ** time - 1) / (rate/100)), 3)
        
growth_recur_dep.iloc[0] = 1
growth_recur_dep_log = np.log(growth_recur_dep).transpose()
labels = growth_recur_dep.transpose()


# plot result

import seaborn as sns
fig, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(260, 0, n=9, as_cmap=True)
ax = sns.heatmap(growth_recur_dep_log, 
                 cmap=cmap, 
                 annot=labels,
                 fmt='.3g', 
                 annot_kws={"fontsize":18},
                 vmin=growth_recur_dep_log[np.isfinite(growth_recur_dep_log)].min().min(),  #a[np.isfinite(a)]
                 vmax=growth_recur_dep_log.max().max(),
                 square=True, 
                 linewidths=.5, 
                 cbar=False)
plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=24, fontname='Helvetica')
plt.setp(ax.get_yticklabels(), rotation=0, ha='right', fontsize=24, fontname='Helvetica')
plt.xlabel('\nTime horizon (y)', fontsize=28, fontname='Helvetica')
plt.ylabel('Return (%)\n', fontsize=28, fontname='Helvetica')
ax.invert_yaxis()
ax.set_facecolor([1,1,1])
ax.set_title('$Investment$ $return$: $recurring$ $deposits$\n', fontsize=32, fontname='Helvetica')

figure_name = './images/10 - hypothetical returns/hypothetical returns - recurring deposits.png'
plt.savefig(figure_name, dpi = 250)
plt.show()

