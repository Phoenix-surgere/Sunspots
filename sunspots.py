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

##Those are imported from a different, dedicated "helper file" but for now I will define them here
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

#series = np.squeeze(series)  #b/c DNN in first layer only works with 1D array w/ TSGenerator - Comment out when LSTM/CNN/ 1st layer
train = series[0:int(len(series)*0.9)] 
test = series[int(len(series)*0.9):]

#Scaling added for NNs
scaler = SS()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# define generator -> Need separate gennie for Validation
n_input = 3  #or window_size, potential hyperparameter 
gentrain = TimeseriesGenerator(series, series, length=n_input, batch_size=1) 
#genval = ...  figure out how to properly set this up
gentest = TimeseriesGenerator(series, series, length=n_input, batch_size=1)

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
    

#NEED BASELINE! ------- Use generators, for even comparisons


#-------- /Baseline resuls 
    
from keras import Input, layers
from keras.models import Model
from keras.optimizers import Adam


#KERAS FUNCTIONAL API PRACTICE 
def MLP(train, test):
    input_tensor = Input(shape=(n_input, ))      
    x = layers.Dense(250, activation='relu')(input_tensor)
    x = layers.Dense(250, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output_tensor = layers.Dense(1)(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=1, 
                    epochs=150,validation_data=test, verbose=1)
    return history

histories = MLP(gentrain, gentest)
plot_metrics(histories)

#ConvNet
def CNN(train, test):
    input_tensor = Input(shape=(n_input, 1))
    x = layers.Conv1D(filters=32, padding='same' , kernel_size=2)(input_tensor)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x) 
    x = layers.Conv1D(filters=32, padding='same' , kernel_size=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(250, activation='relu')(x)
    x =  layers.Dropout(0.25)(x)
    x = layers.Dense(125, activation='relu')(x)
    output_tensor = layers.Dense(1)(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    adam = Adam()
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=10, 
                epochs=50,validation_data=test, verbose=1)
    return history

histories = CNN(gentrain, gentest)
plot_metrics(histories)


#LSTM 
def BiLSTM(train, test):
    #lstm shape: [samples, timesteps, features]
    input_tensor = Input(shape=(n_input, 1))
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(input_tensor)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(16, activation='relu'))(x)
    x = layers.Dropout(0.3)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()
    output_tensor = layers.Dense(1)(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=1, 
                    epochs=5,validation_data=test, verbose=1)
    return history

histories = BiLSTM(gentrain, gentest)
plot_metrics(histories)    

 
def Hybrid(train, test):
    input_tensor = Input(shape=(n_input, 1))
    x = layers.Conv1D(filters=32, padding='same' , kernel_size=2)(input_tensor)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x) 
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(250)(x)
    x = layers.LeakyReLU()(x)
    output_tensor = layers.Dense(1)(x)
    model = Model(input_tensor, output_tensor)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=1, 
            epochs=15,validation_data=test, verbose=1)
    return history

histories = Hybrid(gentrain, gentest)
plot_metrics(histories)    


