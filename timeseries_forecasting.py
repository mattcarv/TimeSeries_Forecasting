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
plt.show()

# Multiplicative residuals look random (there's little seasonality) so we'll use it

#%%
# HOW TO MAKE A TIME SERIES STATIONARY?