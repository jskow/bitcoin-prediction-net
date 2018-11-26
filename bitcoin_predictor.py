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
import postprocess
import os

#Choose verbose, and model type
parser = argparse.ArgumentParser(description="Use -h for options.  Modify model/training parameters.")
parser.add_argument('--model', nargs='?', default=0,
                    help='Type of model (1 for LSTM)',type=int)
parser.add_argument('--plots', nargs='?', default=1,
                    help='Choose to show or hide plots',type=int)
parser.add_argument('--timesteps', nargs='?', default=1,
                    help='Number of timesteps (LSTM)',type=int)
parser.add_argument('--log', nargs='?', default=0,
                    help='Enable/Disable log (1 for enable)',type=int)
#Add parser for # of LSTM units/nodes, # EPOCHS, Learning rate
parser.add_argument('--nn_nodes', nargs='?', default=16,
                    help='Nodes in NN',type=int)
parser.add_argument('--epochs', nargs='?', default=500,
                    help='Number of epochs',type=int)
parser.add_argument('--learning_rate', nargs='?', default=.001,
                    help='learning rate',type=float)
parser.add_argument('--validation_split', nargs='?', default=.2,
                    help='validation split',type=float)
parser.add_argument('--use_optimal', nargs='?', default=0,
                    help='validation split',type=int)

args = parser.parse_args()
#Make this 1 if you want to see some fancy training & error plots
show_plots = args.plots
#Make this 1 if you want to run the LSTM model
do_lstm = args.model
#Create time steps for LSTM
timestep = args.timesteps
do_log = args.log
#New logging args
nn_nodes = args.nn_nodes
EPOCHS = args.epochs
learning_rate = args.learning_rate
validation_split = args.validation_split
use_optimal = args.use_optimal

#If use_optimal, then we use the settings based on overnight testing data
if use_optimal:
    if do_lstm == 1:
        if timestep == 1:
            EPOCHS = 150
            nn_nodes = 144
            learning_rate = .001
            validation_split=0.1
        else:
            timestep = 5
            EPOCHS = 150
            nn_nodes = 16
            learning_rate = .001
            validation_split = 0.2
    else:
        EPOCHS = 400
        nn_nodes = 64
        learning_rate = 0.06
        validation_split = 0.125

if do_lstm == 1:
    print("Running LSTM model, timesteps: ", timestep)
else:
    print("Running basic NN model")
print(nn_nodes, "nodes/layer")
print(EPOCHS, "epochs")
print(learning_rate, "= learning rate")
print(str(validation_split*100) + "%", "= validation_split")


#Specify the dates to test on
datelist = pd.date_range(start='31-12-2011', end='11-10-2018')
num_test_data = 100

#Load all the data sets
morg_vals, morg_test_vals = data.load_morg_avg_us(datelist, num_test_data)
trend_data, trend_test_data = data.load_google_trends_data(datelist, num_test_data)
train_data, train_labels, test_data, test_labels, test_dates = \
                                data.bitcoin_load_data_with_pd(datelist, num_test_data)

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
    model = models.build_basic_model(train_data, learning_rate, nn_nodes)
else:
    if timestep == 1:
        train_data, train_labels, test_data, test_labels, scaler = \
            preprocess.prep_lstm_data(train_data, train_labels, test_data, test_labels,num_test_data)
    else:
        train_data, train_labels, test_data, test_labels, scaler, orig_dataset = \
            preprocess.prep_lstm_data_with_time(train_data, train_labels, test_data, test_labels,num_test_data,timestep)
    model = models.build_lstm_model(train_data, learning_rate, nn_nodes)

#Print structure of model we are using
model.summary()

# Store training stats
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCHS)
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=validation_split, verbose=0,
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
    if timestep == 1:
        test_inverse, prediction_inverse = \
            postprocess.lstm_post_process(test_data, test_predictions, test_labels, scaler)
    else:
        test_inverse, prediction_inverse, test_dates = \
            postprocess.lstm_post_process_with_time(orig_dataset, timestep, test_predictions, datelist, scaler,num_test_data)

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

#Log the data for comparing the models
#If there is no log directory, create one and create a "bitcoin_log.csv" file
if do_log:
    path = os.getcwd()
    if os.path.isdir(path + "/log") != True:
        os.mkdir(path + "/log")

    log = open(path + "/log/bitcoin_log.csv","a+")
    if do_lstm:
        #Save LSTM, # time steps, RMSE, # EPOCHS, # units
        log.write("lstm," + str(timestep) + "," + str(RMSE) + "," + str(EPOCHS) + "," + str(nn_nodes) \
                  + "," + str(learning_rate) + "," + str(validation_split) + "\n")
    else:
        log.write("nn," + str(timestep) + "," + str(RMSE) + "," + str(EPOCHS) + "," + str(nn_nodes) \
                  + "," + str(learning_rate) + "," + str(validation_split) + "\n")

    log.close()
