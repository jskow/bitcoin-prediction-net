#!/usr/local/bin/python3.7
#Postprocess bitcoin training/testing data
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def lstm_post_process(test_data, test_predictions, test_labels, scaler):
    #Reformat the test data so we can get the raw prediction values in $
    test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[2]))
    prediction_data_out = np.column_stack((test_data, test_predictions))
    test_data_out = np.column_stack((test_data, test_labels))
    #Un-scale the prediction and test data
    prediction_inverse = scaler.inverse_transform(prediction_data_out)
    test_inverse = scaler.inverse_transform(test_data_out)
    return test_inverse, prediction_inverse

def lstm_post_process_with_time(orig_dataset, timestep, test_predictions,datelist, scaler, num_test_data):
    #If we have timesteps, get original data from stored matrix
    num_entries = orig_dataset.shape[0]
    test_data = orig_dataset[num_entries-num_test_data:,:-1]
    test_labels = orig_dataset[num_entries-num_test_data:,-1]
    test_data = test_data[:-(timestep-1)]
    test_labels = test_labels[timestep-1:]
    prediction_data_out = np.column_stack((test_data, test_predictions))
    test_data_out = np.column_stack((test_data, test_labels))
    #Un-scale the prediction and test data
    prediction_inverse = scaler.inverse_transform(prediction_data_out)
    test_inverse = scaler.inverse_transform(test_data_out)

    #Reconfigure datelist to account for timestepped data
    test_dates = datelist
    test_dates = test_dates[num_entries-num_test_data:-(timestep-1)]
    return test_inverse, prediction_inverse, test_dates
