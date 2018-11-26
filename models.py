#!/usr/bin/python3.5
#All the bitcoin testing models
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM

#Build a tensorflow model.  2 hidden layers, also train it
def build_basic_model(train_data, learning_rate, nn_nodes):
  model = keras.Sequential([
    keras.layers.Dense(nn_nodes, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(nn_nodes, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

def build_lstm_model(train_data, learning_rate, nn_nodes):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(nn_nodes, return_sequences=True, input_shape=(train_data.shape[1],train_data.shape[2])))
    model.add(keras.layers.LSTM(nn_nodes))
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.Adam(lr=learning_rate)

    # compile and fit the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

#TBD - Add GRU LSTM support
