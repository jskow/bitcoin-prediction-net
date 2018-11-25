#!/usr/local/bin/python3.7
#All the bitcoin testing models
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM

#Build a tensorflow model.  2 hidden layers, also train it
def build_basic_model(train_data):
  model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

def build_lstm_model(train_data):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(16, return_sequences=True, input_shape=(train_data.shape[1],train_data.shape[2])))
    model.add(keras.layers.LSTM(16))
    model.add(keras.layers.Dense(1))

    # compile and fit the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

#TBD - Add GRU LSTM support
