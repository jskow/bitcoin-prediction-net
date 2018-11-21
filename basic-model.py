#!/usr/local/bin/python3.7
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import datetime

#Make this 1 if you want to see some fancy training & error plots
show_plots = 1

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
morg_vals, morg_test_vals = load_morg_avg_us()

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

trend_data, trend_test_data = load_google_trends_data()

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

train_data, train_labels, test_data, test_labels, test_dates = bitcoin_load_data_with_pd()
#Add trend data to market data
train_data = np.column_stack((train_data,trend_data))
test_data = np.column_stack((test_data, trend_test_data))

#Add 15-year US mortgage interest rates
train_data = np.column_stack((train_data,morg_vals))
test_data = np.column_stack((test_data, morg_test_vals))

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

#Normalize the dataset
# Test data is *not* used when calculating the mean and std
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print("First training sample, normalized:")
print(train_data[0])  # First training sample, normalized

#Build a tensorflow model.  2 hidden layers, also train it
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

#Number of times to train on all the data
EPOCHS = 500

# Store training stats
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

#Visualize the training data
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [$1]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 2000])
  if show_plots == 1:
      plt.show()

plot_history(history)

#Test on test data set
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae))


#Predict on data set
test_predictions = model.predict(test_data).flatten()

#Plot test value and test prediction versus days
x_label = test_dates[0].strftime("%B %d, %Y") + ' ... ' + test_dates[99].strftime("%B %d, %Y")
print(x_label)
date_idx = np.arange(0,100,1)
plt.scatter(date_idx, test_predictions)
plt.scatter(date_idx, test_labels)
plt.xlabel(x_label)
plt.ylabel('Predictions [$1]')
#plt.axis('equal')
plt.xlim([-5,105])
plt.ylim(plt.ylim())
_ = plt.plot(date_idx, test_predictions, label='Predictions')
plt.plot(date_idx, test_labels, label='Actual')
plt.legend(loc='upper right')
if show_plots == 1:
    plt.show()

#Error of data set
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [$1]")
_ = plt.ylabel("Count")
if show_plots == 1:
    plt.show()