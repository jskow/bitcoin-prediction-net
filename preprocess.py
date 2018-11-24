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

def prep_lstm_data(train_data, train_labels, test_data, test_labels):
    dataset = np.column_stack((train_data,train_labels))
    test_data_lstm = np.column_stack((test_data, test_labels))
    dataset = np.vstack((dataset, test_data_lstm))
    #Normalize the dataset to 0,1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    #Extract training data/labels
    num_entries = dataset.shape[0]
    train_data = dataset[:num_entries-100,:-1]
    train_labels = dataset[:num_entries-100,-1]
    test_data = dataset[num_entries-100:,:-1]
    test_labels = dataset[num_entries-100:,-1]

    #Format for LSTM layer (batch, time steps, # features)
    train_data = np.reshape(train_data, (len(train_data), 1, train_data.shape[1]))
    test_data = np.reshape(test_data, (len(test_data), 1, test_data.shape[1]))
    return train_data, train_labels, test_data, test_labels, scaler
