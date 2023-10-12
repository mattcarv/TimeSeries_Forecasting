import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


path = '/home/mdocarm/Downloads/archive(1)/AirPassengers.csv'

df = pd.read_csv(path)

df.columns = ['Date', 'Number_of_Passengers']
#df.head()

plt.figure(figsize=(15, 4))
plt.plot(df['Date'], df['Number_of_Passengers'])
plt.xticks(['1949-12', '1951-12', '1953-12', '1955-12', '1957-12', '1959-12'])
plt.title('Number of passengers from 1949 to 1960')
plt.xlabel('Year-Month')
plt.ylabel('Number')
plt.clf()


#%%
# DECOMPOSITION OF A TIME SERIES

from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# do we want an additive or multiplicative approach?

mult = seasonal_decompose(df['Number_of_Passengers'], model='multiplicative', period=30)
add = seasonal_decompose(df['Number_of_Passengers'], model='additive', period=30)


mult.plot().suptitle('Multiplicative')
add.plot().suptitle('Additive')
plt.clf()

# Multiplicative residuals look random (there's little seasonality) so we'll use it

#%%
# HOW TO MAKE A TIME SERIES STATIONARY?
# The statistical properties of the series like mean, variance and 
# autocorrelation are constant over time.

# Autoregressive forecasting models are essentially linear regression models 
# that utilize the lag(s) of the series itself as predictors.

# 
   # Differencing the Series (once or more)
   # Take the log of the series
   # Take the nth root of the series
   # Combination of the above
#
#%%
# Detrend a time series

# Using scipy: subtract the line of best fit
from scipy import signal

detrend = signal.detrend(df['Number_of_Passengers'].values)

plt.plot(detrend)
plt.title('Detrended by the least squares fit')
plt.clf()

# Using statmodels: subtracting the trend component
from statsmodels.tsa.seasonal import seasonal_decompose

detrend = df['Number_of_Passengers'].values - mult.trend

plt.plot(detrend)
plt.title('Detrended by subtracting the trend component')
plt.clf()
#
#%%
# Deseasonalize a time series

deseason = df['Number_of_Passengers'].values / mult.seasonal

plt.plot(deseason)
plt.title('Deseasonalized')
plt.clf()

#
#%%
# Test the seasonality of a time series

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df['Number_of_Passengers'].tolist())

plt.clf()

#%%
# Autocorrelation Function and Partial Autocorrelation Function
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(df['Number_of_Passengers'].tolist(), lags=60, ax=ax[0])
plot_pacf(df['Number_of_Passengers'].tolist(), lags=60, ax=ax[1], auto_ylims=True)
plt.clf()

#%%

# Lag Plots

from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(df['Number_of_Passengers'], lag=i+10, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+10))

fig.suptitle('Lag Plots of Air Passengers', y=1.1)    
plt.clf()

#%%

# Granger Causality test

from statsmodels.tsa.stattools import grangercausalitytests

data = pd.read_csv('/home/mdocarm/Downloads/archive(1)/dataset.txt')
data['date'] = pd.to_datetime(data['date'])
data['month'] = data.date.dt.month
grangercausalitytests(data[['value', 'month']], maxlag=5)

# p-value being zero for all tests shows that the months can be used to 
# forecast the time series
