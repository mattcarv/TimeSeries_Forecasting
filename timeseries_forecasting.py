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

plt.show()
