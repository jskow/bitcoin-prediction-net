#!/usr/local/bin/python3.7
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import datetime
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from math import sqrt
import argparse

#All models defined in models.py
import models
#All data loading functions
import data
#Preprocess the data
import preprocess
#Display training process/prediction results
import display

#Choose verbose, and model type
parser = argparse.ArgumentParser(description="Choose model type/plot visibility. \
                                            ./bitcoin_predictor 1 1 is LSTM with \
                                            visible plots.")
parser.add_argument('model', nargs='?', default=0,
                    help='Type of model (1 for LSTM)',type=int)
parser.add_argument('plots', nargs='?', default=1,
                    help='Choose to show or hide plots',type=int)

args = parser.parse_args()
#Make this 1 if you want to see some fancy training & error plots
show_plots = args.plots
#Make this 1 if you want to run the LSTM model
do_lstm = args.model
#Number of times to train on all the data
EPOCHS = 500
if do_lstm == 1:
    print("Running LSTM model")
else:
    print("Running basic NN model")

#Load all the data sets
morg_vals, morg_test_vals = data.load_morg_avg_us()
trend_data, trend_test_data = data.load_google_trends_data()
train_data, train_labels, test_data, test_labels, test_dates = data.bitcoin_load_data_with_pd()

#Add trend data to market data
train_data = np.column_stack((train_data,trend_data))
test_data = np.column_stack((test_data, trend_test_data))

#Add 15-year US mortgage interest rates
train_data = np.column_stack((train_data,morg_vals))
test_data = np.column_stack((test_data, morg_test_vals))

#Preprocess data before submitting to NN
# Shuffle the training set for NN, not for LSTM where order matters
if do_lstm != 1:
    train_data, train_labels, test_data = \
        preprocess.prep_basic_data(train_data, train_labels, test_data)
    model = models.build_basic_model(train_data)
else:
    train_data, train_labels, test_data, test_labels, scaler = \
        preprocess.prep_lstm_data(train_data, train_labels, test_data, test_labels)
    model = models.build_lstm_model(train_data)

#Print structure of model we are using
model.summary()

# Store training stats
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, display.PrintDot()])

#Show training results & check error on data set
if do_lstm:
    if show_plots:
        display.plot_lstm_history(history)
    loss = model.evaluate(test_data, test_labels, verbose=0)
else:
    if show_plots:
        display.plot_history(history)
    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: ${:7.2f}".format(mae))

#Predict on data set & caclulate the root mean square error
if do_lstm != 1:
    test_predictions = model.predict(test_data).flatten()
    RMSE = sqrt(mean_squared_error(test_labels, test_predictions))
    print('Test RMSE: $%.3f' % RMSE)
else:
    #Get matrix to calculate RMSE
    test_predictions = model.predict(test_data)
    #Reformat the test data so we can get the raw prediction values in $
    test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[2]))
    prediction_data_out = np.column_stack((test_data, test_predictions))
    test_data_out = np.column_stack((test_data, test_labels))
    #Un-scale the prediction and test data
    prediction_inverse = scaler.inverse_transform(prediction_data_out)
    test_inverse = scaler.inverse_transform(test_data_out)
    #Caculate root mean square error for predictions
    RMSE = sqrt(mean_squared_error(test_inverse[:,-1], prediction_inverse[:,-1]))
    test_predictions = prediction_inverse[:,-1]
    test_labels = test_inverse[:,-1]
    print('Test RMSE: $%.3f' % RMSE)

#Plot test value and test prediction versus days
if show_plots:
    display.plot_predictions(test_dates, test_predictions, test_labels)

#Error of data set
if show_plots:
    display.plot_prediction_error(test_predictions,test_labels)
