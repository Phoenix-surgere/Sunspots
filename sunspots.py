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
from helper_funcs import adfuller_test, kpss_test,jb_test
#from statsmodels.stats.stattools import jarque_bera
#from tensorflow import keras
#tf.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()

sunspots = pd.read_csv('Sunspots.csv', index_col='Date', parse_dates=True,
                       usecols=['Date', 'Monthly Mean Total Sunspot Number'])
sunspots.rename(columns={'Monthly Mean Total Sunspot Number': 'Sunspots'}, 
                 inplace=True)

#sunspots['year'] = sunspots.index.year
#fig, (ax1,ax2) = plt.subplots(nrows=2)
#sns.lineplot(ax=ax1, x='Date',y='Sunspots',data=sunspots.reset_index())
#sns.lineplot(ax=ax2, x='year',y='Sunspots',data=sunspots)
#ax2.set_xlim(1800, 1850)
#plt.show()



print(adfuller_test(sunspots))
print('*'*5)
print(kpss_test(sunspots))
print('*'*5)
print(jb_test(sunspots))


#sunspot = np.array(sunspots)
#series = tf.expand_dims(sunspot, axis=-1)
#ds = tf.data.Dataset.from_tensor_slices(series)
#window_size = 5
#count = 0
#ds = ds.window(5, shift=1)
##for window_dataset in ds:
##    for val in window_dataset:
##        print(val.numpy(), end=" ")
##    print()
#
#a = np.arange(20)
#adf = pd.DataFrame(a)
#adf.columns = ['x']
#adf['y'] = adf.shift(-1).dropna()
#
#from sklearn.model_selection import TimeSeriesSplit
#tscv = TimeSeriesSplit()
#from keras.preprocessing.sequence import TimeseriesGenerator as TSG
#window_size = 5
#a = a.reshape((len(a), 1))
#gennie = TSG(a, a, length=window_size,batch_size=5)
##for i in range(len(gennie)):
##	x, y = gennie[i]
##	print('%s => %s' % (x, y))
##    
##X,y = adf['x'], adf['y']
##for train_index, test_index in tscv.split(X):
##    print("TRAIN:", train_index, "TEST:", test_index)
##    X_train, X_test = X[train_index], X[test_index]
##    y_train, y_test = y[train_index], y[test_index]
##model = tf.keras.models.Sequential([
##        tf.keras.layers.Dense(1, input_shape=window_size )
##              ])
##model.compile(loss='mse', optimizer='adam')
##model.fit_generator(gennie, steps_per_epoch=1, epochs=2, verbose=0)
##
#def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
#    series = tf.expand_dims(series, axis=-1)
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
#    ds = ds.shuffle(shuffle_buffer)
#    ds = ds.map(lambda w: (w[:-1], w[1:]))
#    return ds.batch(batch_size).prefetch(1)
#
##train_set = windowed_dataset(a, 4,2,10)
#
#from keras.models import Sequential
#from keras.layers import Dense, Flatten, Conv1D
#model = Sequential()
#model.add(Conv1D(10, kernel_size=3, input_shape=(window_size, 1),padding='valid'))
##model.add(LSTM(10, input_shape=[window_size,1]))
#model.add(Flatten())
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
#history = model.fit_generator(gennie, epochs=10)