import numpy as np # linear algebra
from numpy.random import seed 
import math 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, date 

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import seaborn as sns

import warnings # Supress warnings 
warnings.filterwarnings('ignore')

import statsmodels as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA


# Using stattools to check if the time series is stationary or not

from statsmodels.tsa.stattools import adfuller

def check_stat(series):
    res = adfuller(series.values)
    
    print('ADF Statistic: ', res[0])
    print('p-value: ', res[1])
    print('Critical Values: ')
    for key, value in res[4].items():
        print('\t%s: %.3f' % (key, value))
        
    if (res[1] <= 0.05) & (res[4]['5%'] > res[0]):
        print('Stationary')
    else:
        print('Non-stationary')
        
        
# --Modelling the time series using an Auto-regressive model

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

## HI MATT, ITS YOU FROM THE FUTURE... BE CAREFUL!!
## GOODBYE FOR NOW.....

alpha1 = 0.5
nsamples = 100
seed = 1
np.random.seed(seed)

ar = np.r_[1, -np.array([alpha1])]
ma = np.r_[1]

timeseries = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=nsamples, freq='MS'),
                           't': sm.tsa.arima_process.arma_generate_sample(ar, ma, nsamples)})

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(timeseries.timestamp, timeseries['t'], marker='o')
plt.title('Timeseries')
plt.xlabel('Time')
plt.clf()

# --Let's check if this time series is stationary

# check_stat(timeseries['t'])

# --Now we can see if the series has total or partial autocorrelation

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(timeseries['t'], lags=15, ax=ax[0], auto_ylims=True)
plot_pacf(timeseries['t'], lags=15, ax=ax[1], auto_ylims=True)
plt.tight_layout()
plt.clf()

# Here we see a strong correlation at lag 1
# We will, now, model and predict this time series using an AutoRegressor

train_len = int(0.8*nsamples)
train = timeseries['t'][:train_len]
ar_model = AutoReg(train, lags=1).fit()
# print(ar_model.summary())

# This gives us a coef of 0.42, not too far from alpha1 = 0.5
# Now we plot the predicted values from this model

pred = ar_model.predict(start=train_len, end=nsamples-1, dynamic=False)


fig, ax = plt.subplots(1, 1, figsize=(15, 4))
plt.plot(timeseries.timestamp[train_len:nsamples], timeseries.t[train_len:nsamples],
          marker='o', c='grey', label='Test')
plt.plot(timeseries.timestamp[:train_len], train, marker='o',
          c='k', label='Train')
plt.plot(timeseries.timestamp[train_len:nsamples], pred, marker='o', c='red',
         label='Prediction')
plt.title('Timeseries')
plt.xlabel('Time')
plt.legend()
plt.show()