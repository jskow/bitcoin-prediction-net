#!/usr/local/bin/python3.7
#Load bitcoin training/testing data
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM
import pandas as pd

#Load 15 year mortagage average interest rate
#15-Year Fixed Rate Mortgage Average in the United States
def load_morg_avg_us():
    data = pd.read_csv('bitcoin-historical-data/MORTGAGE15US.csv',parse_dates=True,index_col=0)

    #Create data with all dates we want
    datelist = pd.date_range(start='31-12-2011', end='11-10-2018')
    #Reindex based on days, and fill the new days based on the previous week data
    data = data.reindex(datelist, method="ffill")

    #Remove last data since we don't know the future
    data = data[:(len(data)-1)]

    #Record number of samples
    num_entries = len(data)

    #Set aside 100 values for validation
    test_data = data[num_entries-100:]
    data = data[:num_entries-101]

    return data.values, test_data.values


#Create dataset from weekly google trends dataset
#Market data starts 12-31-2011, so only start data from there
def load_google_trends_data():
    data = pd.read_csv('bitcoin-historical-data/bitcoin_search_volume.csv',parse_dates=True,index_col=0)

    #Create data with all dates we want
    datelist = pd.date_range(start='31-12-2011', end='11-10-2018')
    #Reindex based on days, and fill the new days based on the previous week data
    data = data.reindex(datelist, method="ffill")

    #Remove last data since we don't know the future
    data = data[:(len(data)-1)]

    #Record number of samples
    num_entries = len(data)

    #Set aside 100 values for validation
    test_data = data[num_entries-100:]
    data = data[:num_entries-101]

    return data.values, test_data.values

#Get bitcoin dataset
def bitcoin_load_data_with_pd():
    data = pd.read_csv('bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv')

    #Group the data by day instead of by minute
    data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date
    group = data.groupby('date')
    Daily_Price = group['Weighted_Price'].mean()

    #Drop any row which as NaN values
    data = data.dropna(axis=0)

    #Ensure only 1 entry/day
    data = data.drop_duplicates(subset='date')

    #Check with rows have NaN so we can filter other data also
    #Make sure all dates are present.  Find missing days and scrub from all data
    datelist = pd.date_range(start='31-12-2011', end='11-10-2018').date
    #Index the data by date
    data = data.set_index('date')
    #Duplicate data to fill for any missing entries
    data = data.reindex(datelist, method="ffill")
    Daily_Price = Daily_Price.reindex(datelist, method="ffill")

    #Remove the 1st daily price, since we want the 1st day to predict 2nd day price
    Daily_Price = Daily_Price[1:]
    #Remove last data point, since we don't know what tomorrow's price is
    data = data[:(len(data)-1)]

    #Record number of samples
    num_entries = len(Daily_Price)

    #Collect testing dates for plotting purposes
    testing_dates = datelist
    testing_dates = testing_dates[num_entries-100:]

    #Drop timestamp, date from dataset
    data = data.drop(labels='Timestamp', axis=1)

    #Set aside 100 entires for testing
    test_data = data[num_entries-100:]
    test_prices = Daily_Price[num_entries-100:]
    data = data[:num_entries-101]
    Daily_Price = Daily_Price[:num_entries-101:]

    #Get final matrix form without the indices
    data = data.values
    Daily_Price = Daily_Price.values
    test_data = test_data.values
    test_prices = test_prices.values

    return data, Daily_Price, test_data, test_prices, testing_dates
