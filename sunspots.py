# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:17:22 2019

@author: black
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

##Those are imported from a different, dedicated "helper file" but for now I will let them here
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera
#define ADF     
def adfuller_test(timeseries):
    print('H0: Unit Root (Non-Stationary)')
    print('Results of ADF Test: ')
    timeseries = timeseries.iloc[:, 0].values
    adftest = adfuller(timeseries)
    adfout = pd.Series(adftest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key, value in adftest[4].items():
        adfout['Critical Value (%s)'%key] = value
    return adfout
#replaced print (adfout, kpss_output) with return to avoid none return

#define KPSS
def kpss_test(timeseries):
    print('H0: Stationarity')
    print ('Results of KPSS Test:')
    timeseries = timeseries.iloc[:, 0].values
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    return kpss_output

#define Jarque-Bera
def jb_test(timeseries):
    #large JB-Value: NOT normal distribution
    print('H0: Normal Distribution')
    print ('Results of J-B Test:')
    timeseries = timeseries.iloc[:, 0].values
    jbtest = jarque_bera(timeseries)
    jb_output = pd.Series(jbtest, index=['Test Statistic','p-value',
        'Skeweness', 'Kurtosis'])
    return jb_output

  
sunspots = pd.read_csv('Sunspots.csv', index_col='Date', parse_dates=True,
                       usecols=['Date', 'Monthly Mean Total Sunspot Number'])
sunspots.rename(columns={'Monthly Mean Total Sunspot Number': 'Sunspots'}, 
                 inplace=True)

#An initial plot of the data, focusing on one random subset to look for any trends (subject to change later)
sunspots['year'] = sunspots.index.year
fig, (ax1,ax2) = plt.subplots(nrows=2)
sns.lineplot(ax=ax1, x='Date',y='Sunspots',data=sunspots.reset_index())
sns.lineplot(ax=ax2, x='year',y='Sunspots',data=sunspots)
ax2.set_xlim(1800, 1850)
plt.show()

#More visuals
sns.distplot(sunspots, bins=75, kde=False)
plt.xlabel('Monthly no of Sunspots (median)')
plt.ylabel('Frequency of Monthly Mean no of Sunspots')
plt.show()

sns.kdeplot(sunspots['Sunspots'], shade=True)
plt.xlabel('Monthly no of Sunspots (median)')
plt.ylabel('Density of Monthly Mean no of Sunspots')
plt.show()

#ADF and KPSS agree our series is stationary, although JB says it is not normally distributed.
print(adfuller_test(sunspots))
print('*'*5)
print(kpss_test(sunspots))
print('*'*5)
print(jb_test(sunspots))


#MODELLING: Problem framed as univariate one-step ahead predict problem - could experiment with prediction horizon
from numpy import array
from keras.preprocessing.sequence import TimeseriesGenerator
series = array(sunspots)  #need split into train - test

series = np.squeeze(series)  #because DNNs in first layer only work with 1D array w/ TSGenerator - Comment out when LSTM/CNN/...
train = series[0:int(len(series)*0.9)] 
test = series[int(len(series)*0.9):]


# define generator
n_input = 3  #or window_size, variable, potential hyperparameter 
generatortr = TimeseriesGenerator(series, series, length=n_input, batch_size=1)  #verbose?
generatorts = TimeseriesGenerator(series, series, length=n_input, batch_size=1)

# number of samples + visual check, script taken from ml mastery - uncomment to confirm
#print('Samples: %d' % len(generatorts))
# print each sample
#for i in range(len(generatorts)):
#	x, y = generatorts[i]
#	print('%s => %s' % (x, y))


#lstm: [samples, timesteps, features]

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, LSTM, Dropout

#TO DO: NEED SCALING!!!!! Also potential hyper (eg normalize or scale ~ (0,1) ? 

#1st NN used, simplistic but still good for practice
model = Sequential()
model.add(Dense(250, activation='relu', input_dim=n_input))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
history = model.fit_generator(generatortr, steps_per_epoch=1, 
                              epochs=30, validation_data=generatorts)

mae = history.history['mean_absolute_error']
mae_val = history.history['val_mean_absolute_error']

loss = history.history['loss']
loss_val = history.history['val_loss']

epochs = range(len(mae))
plt.plot(epochs, mae);plt.plot(epochs, mae_val) 
plt.title('Train and Val MAE'); plt.show()

plt.plot(epochs, loss);plt.plot(epochs, loss_val) 
plt.title('Train and Val Loss'); plt.show()

#TO DO: More models, this repo will focus on NNs
