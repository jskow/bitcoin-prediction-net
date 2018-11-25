#!/usr/local/bin/python3.7
#Display utility functions for plotting output data
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

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
  plt.show()

def plot_lstm_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Loss [$1]')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
  plt.legend()
  plt.show()

def plot_predictions(test_dates, test_predictions, test_labels):
    x_label = test_dates[0].strftime("%B %d, %Y") + ' ... ' + test_dates[len(test_dates)-1].strftime("%B %d, %Y")
    date_idx = np.arange(0,len(test_dates),1)
    plt.scatter(date_idx, test_predictions)
    plt.scatter(date_idx, test_labels)
    plt.xlabel(x_label)
    plt.ylabel('Predictions [$1]')
    plt.xlim([-5,len(test_dates)+5])
    plt.ylim(plt.ylim())
    plt.plot(date_idx, test_predictions, label='Predictions')
    plt.plot(date_idx, test_labels, label='Actual')
    plt.legend(loc='upper right')
    plt.show()

def plot_prediction_error(test_predictions,test_labels):
    error = test_predictions - test_labels
    plt.hist(error, bins = 50)
    plt.xlabel("Prediction Error [$1]")
    plt.ylabel("Count")
    plt.show()
