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
#SCALING:
train_size = int(len(series) * 0.8)
val_size = int(len(series) * 0.1)

scaler = SS()
scaler.fit(series[0:train_size])  #fit only on train data...
series = scaler.transform(series)  #transform the entire data based on train

#If running DNN, have this line commented
series = np.squeeze(series)

# Need 3 generators (train, val, test) - see chollet pg 235 (ebook page)
n_input = 3  #or whichever window_size preferred

gentrain = TSG(series, series, length=n_input, start_index=0,
                               end_index=train_size, batch_size=1)

genval = TSG(series, series, length=n_input, start_index = train_size+1,
                             batch_size=1, end_index=train_size+val_size)

gentest = TSG(series, series, length=n_input, start_index = train_size+val_size+1,
              batch_size=1, end_index=None)


def plot_metrics(history):

    mae = history.history['mean_absolute_error']
    mae_val = history.history['val_mean_absolute_error']
    epochs = range(len(mae))
    
    loss = history.history['loss']
    loss_val = history.history['val_loss']
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7,7))

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
#Naive Baseline
scores = {}
from sklearn.metrics import mean_squared_error as mse
series_naive = pd.DataFrame(series)
series_naive['naive_pred'] = series_naive.shift(1)
series_naive.dropna(inplace=True)

#Train cv and test MSEs - inverse transformed RMSE reported
#May not actually need cv, but will keep it for now to be consistent
naive_mse_tr = mse(series_naive.iloc[0:train_size,0], 
                   series_naive['naive_pred'].iloc[0:train_size])

naive_mse_vs = mse(series_naive.iloc[train_size+1:train_size+val_size ,0], 
                   series_naive['naive_pred'].iloc[train_size+1:train_size+val_size])

naive_mse_ts = mse(series_naive.iloc[train_size+val_size+1: ,0], 
                   series_naive['naive_pred'].iloc[train_size+val_size+1:])

#Unscaled and on same scale on each set
def append_scores(model, tr, cv, ts):
    scores[f"{model}_train_RMSE"] =  float(scaler.inverse_transform([tr])**0.5)
    scores[f"{model}_cv_RMSE"] =   float(scaler.inverse_transform([cv])**0.5)
    scores[f"{model}_test_RMSE"] =  float(scaler.inverse_transform([ts])**0.5)
    return scores

append_scores("NaiveBaseline", naive_mse_tr, naive_mse_vs, naive_mse_ts)
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

histories = MLP(gentrain, genval)
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

histories = CNN(gentrain, genval)
plot_metrics(histories)


#LSTM 
def BiLSTM(train, test):
    #lstm shape: [samples, timesteps, features]
    input_tensor = Input(shape=(n_input, 1))
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(input_tensor)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(16, activation='relu', dropout=0.2, recurrent_dropout=0.2))(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()
    output_tensor = layers.Dense(1)(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=1, 
                    epochs=5,validation_data=test, verbose=1)
    return history

histories = BiLSTM(gentrain, genval)
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
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(250)(x)
    x = layers.LeakyReLU()(x)
    output_tensor = layers.Dense(1)(x)
    model = Model(input_tensor, output_tensor)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit_generator(train, steps_per_epoch=1, 
            epochs=15,validation_data=test, verbose=1)
    return history

histories = Hybrid(gentrain, genval)
plot_metrics(histories)    


