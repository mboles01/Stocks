# # Historical S&P 500 analysis

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Stocks/') 

# load packages
import pandas as pd
import numpy as np

## IMPORT AND COMBINE DATA SETS



# S&P 500 daily closing value 1986 - 2018
sp_old_raw = pd.read_csv('./data/sp500_to_29jun2018.csv')

sp_new_raw_temp1 = pd.read_csv('./data/sp500_to_9apr2020.csv')
sp_new_raw_temp2 = sp_new_raw_temp1[['Effective date ', 'S&P 500']]
sp_new_raw = sp_new_raw_temp2[sp_new_raw_temp2['Effective date '].notna()]

sp_old = pd.DataFrame({'date': pd.to_datetime(sp_old_raw['date']),
                        'close': sp_old_raw['close']})

sp_new = pd.DataFrame({'date': pd.to_datetime(sp_new_raw['Effective date ']),
                        'close': sp_new_raw['S&P 500']})

# combine old and new data sets
sp = pd.concat([sp_old, sp_new]).drop_duplicates().reset_index(drop=True)




# S&P 500 daily closing value 1927 - 2020
sp_old_raw = pd.read_csv('./data/SP.csv')

# # recast date as datetime
# sp_old = pd.DataFrame({'date': pd.to_datetime(sp_old_raw['Date']),
#                         'close': sp_old_raw['Close']})

sp = sp_old_raw

# save combined data set
sp.to_csv('./data/sp500_all.csv', index=False)


# Load combined S&P 500 data set
sp = pd.read_csv('./data/sp500_all.csv')
sp['date'] = pd.to_datetime(sp['Date'])
sp['close'] = sp['Close']
sp = sp[['date', 'close']]
sp['ln close'] = np.log(sp['close'])
sp[::1000]

