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

acf = plot_acf(sunspots, lags=15, alpha=0.5)
plt.show(acf)
    
pacf = plot_pacf(sunspots, lags=15, alpha=0.5)
plt.show(pacf)

#MODELLING: Problem framed as univariate one-step ahead predict problem - could experiment with prediction horizon
from numpy import array
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler as SS

series = array(sunspots) 

#series = np.squeeze(series)  #because DNNs in first layer only work with 1D array w/ TSGenerator - Comment out when LSTM/CNN/ 1st layer
train = series[0:int(len(series)*0.9)] 
test = series[int(len(series)*0.9):]

#Scaling added for NNs
scaler = SS()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# define generator
n_input = 3  #or window_size, potential hyperparameter 
gentrain = TimeseriesGenerator(series, series, length=n_input, batch_size=1)  #verbose?
gentest = TimeseriesGenerator(series, series, length=n_input, batch_size=1)

#lstm: [samples, timesteps, features]

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, LSTM, Dropout

def dnn(train, test):    
    model = Sequential()
    model.add(Dense(250, activation='relu', input_dim=n_input))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    histories = model.fit_generator(train, steps_per_epoch=1, 
                verbose=0 ,epochs=150, validation_data=test)
    return histories 

def plot_metrics(history):

    mae = history.history['mean_absolute_error']
    mae_val = history.history['val_mean_absolute_error']
    epochs = range(len(mae))
    
    loss = history.history['loss']
    loss_val = history.history['val_loss']
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7,7))
    #fig.suptitle('Loss and Metrics History')

    ax1.title.set_text('Metrics (MAE) History')
    ax2.title.set_text('Loss (MSE) History')
    
    ax1.set_xlabel('Epochs'); ax2.set_xlabel('Epochs')
    
    ax1.plot(epochs, mae, label='Train') 
    ax1.plot(epochs, mae_val, label='Validation')
    ax1.legend(loc='best')
    
    ax2.plot(epochs, loss, label='Train')
    ax2.plot(epochs, loss_val, label='Validation') 
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

#histories = dnn(gentrain, gentest)  #to run, uncomment squeeze from above, but cannot (yet) run the others
#plot_metrics(histories)

def CNN(train, test):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, input_shape=(n_input, 1),
                     padding='valid', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=1, 
                epochs=100,validation_data=test, verbose=0)
    return history

histories = CNN(gentrain, gentest)
plot_metrics(histories)

def BiLSTM(train, test):
    model = Sequential()
    model.add(LSTM(32, input_shape=(n_input,1), return_sequences=True, activation='relu'))
    model.add(Bidirectional(LSTM(32, activation='relu')))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=1, 
                epochs=75,validation_data=test, verbose=1)
    return history

histories = BiLSTM(gentrain, gentest)
plot_metrics(histories)

def Hybrid(train,test):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, input_shape=(n_input, 1),
                     padding='valid', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(64, activation='relu')))
    model.add(Dense(250))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=1, 
            epochs=75,validation_data=test, verbose=1)
    return history

histories = BiLSTM(gentrain, gentest)
plot_metrics(histories)

