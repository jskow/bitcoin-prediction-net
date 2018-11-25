#!/usr/local/bin/python3.7
#Preprocess bitcoin training/testing data
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def prep_basic_data(train_data, train_labels, test_data):
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]
    #Subtract mean divide by std dev to normalize data
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    return train_data, train_labels, test_data

def prep_lstm_data(train_data, train_labels, test_data, test_labels,num_test_data):
    dataset = np.column_stack((train_data,train_labels))
    test_data_lstm = np.column_stack((test_data, test_labels))
    dataset = np.vstack((dataset, test_data_lstm))
    #Normalize the dataset to 0,1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    #Extract training data/labels
    num_entries = dataset.shape[0]
    train_data = dataset[:num_entries-num_test_data,:-1]
    train_labels = dataset[:num_entries-num_test_data,-1]
    test_data = dataset[num_entries-num_test_data:,:-1]
    test_labels = dataset[num_entries-num_test_data:,-1]


    #Format for LSTM layer (batch, time steps, # features)
    train_data = np.reshape(train_data, (len(train_data), 1, train_data.shape[1]))
    test_data = np.reshape(test_data, (len(test_data), 1, test_data.shape[1]))
    return train_data, train_labels, test_data, test_labels, scaler

def prep_lstm_data_with_time(train_data, train_labels, test_data, test_labels,num_test_data,timesteps):
    #Default is only have 1 time step for 1 feature
    #If we want to change the # of timesteps, we need to re-arrange the data

    #Ex: with 1 time step and 100 examples, we have size 100,1,8 (8 features)
    #With 5 time steps and 100 examples, we have size 100,5,8
    #So for each time step, we need to take the previous 4 also
    #If num_test_data is 100, then we will take from num_data-104 to num_data-100 for sample 1

    #first normalize the data
    dataset = np.column_stack((train_data,train_labels))
    test_data_lstm = np.column_stack((test_data, test_labels))
    dataset = np.vstack((dataset, test_data_lstm))

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    #Extract training data/labels
    num_entries = dataset.shape[0]
    train_data = dataset[:num_entries-num_test_data,:-1]
    train_labels = dataset[:num_entries-num_test_data,-1]
    test_data = dataset[num_entries-num_test_data:,:-1]
    test_labels = dataset[num_entries-num_test_data:,-1]

    #Save the original scaled dataset
    orig_dataset = dataset

    #Now re-create the training data
    #Create new train data that has the correct number of values
    #Ex: if we want 2 timesteps, we want to go from index 1 to end - 1
    new_train_data = train_data[:-(timesteps-1)]
    new_test_data = test_data[:-(timesteps-1)]
    train_col_size = new_train_data.shape[0]
    test_col_size = new_test_data.shape[0]

    #Each new column should start from i, go to the adjusted column length + i
    #Only want add run timesteps - 1 columns
    for i in range(1,timesteps):
        #Stack the shifted data for each timestep
        #Get all rows from day-timestep, and column stack them
        new_train_data = np.column_stack((new_train_data, train_data[i:(train_col_size+i)]))
        new_test_data = np.column_stack((new_test_data, test_data[i:(test_col_size+i)]))
    #reshape the data for LSTM
    train_data = np.reshape(new_train_data, (len(new_train_data), timesteps, train_data.shape[1]))
    train_labels = train_labels[timesteps-1:]
    test_data = np.reshape(new_test_data, (len(new_test_data), timesteps, test_data.shape[1]))
    test_labels = test_labels[timesteps-1:]

    return train_data, train_labels, test_data, test_labels, scaler, orig_dataset
