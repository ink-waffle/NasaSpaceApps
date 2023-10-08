import os
import math
import json
import re
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import joblib
import numpy as np
import pandas as pd
import random
from keybert import KeyBERT
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
import tensorflow as tf
import datetime
from tensorflow.keras.models import load_model
from tensorflow import keras
from keras import layers
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import gc


X_df_16 = pd.read_csv('Datasets/dsc_fc_summed_spectra_2016_v01.csv', delimiter = ',', parse_dates=[0], infer_datetime_format=True, header = None)
X_df_16.fillna(0,inplace=True)
X_df_17 = pd.read_csv('Datasets/dsc_fc_summed_spectra_2017_v01.csv', delimiter = ',', parse_dates=[0], infer_datetime_format=True, na_values='0', header = None)
X_df_17.fillna(0,inplace=True)
X_df_18 = pd.read_csv('Datasets/dsc_fc_summed_spectra_2018_v01.csv', delimiter = ',', parse_dates=[0], infer_datetime_format=True, na_values='0', header = None)
X_df_18.fillna(0,inplace=True)
X_df_19 = pd.read_csv('Datasets/dsc_fc_summed_spectra_2019_v01.csv', delimiter = ',', parse_dates=[0], infer_datetime_format=True, na_values='0', header = None)
X_df_19.fillna(0,inplace=True)
X_df_20 = pd.read_csv('Datasets/dsc_fc_summed_spectra_2020_v01.csv', delimiter = ',', parse_dates=[0], infer_datetime_format=True, na_values='0', header = None)
X_df_20.fillna(0,inplace=True)
X_df_21 = pd.read_csv('Datasets/dsc_fc_summed_spectra_2021_v01.csv', delimiter = ',', parse_dates=[0], infer_datetime_format=True, na_values='0', header = None)
X_df_21.fillna(0,inplace=True)
X_df_22 = pd.read_csv('Datasets/dsc_fc_summed_spectra_2022_v01.csv', delimiter = ',', parse_dates=[0], infer_datetime_format=True, na_values='0', header = None)
X_df_22.fillna(0,inplace=True)
X_df_23 = pd.read_csv('Datasets/dsc_fc_summed_spectra_2023_v01.csv', delimiter = ',', parse_dates=[0], infer_datetime_format=True, na_values='0', header = None)
X_df_23.fillna(0,inplace=True)
X_df = [X_df_16, X_df_17, X_df_18, X_df_19, X_df_20, X_df_21]

Y_df = []
for i in range(len(X_df)):
    with open('Datasets/DGD/' + str(2016 + i) + '_DGD.txt', 'r') as file:
        lines = file.readlines()[12:]
    days = X_df[i][0].apply(lambda x: " ".join([x.strftime('%Y'), x.strftime('%m'), x.strftime('%d')])).values
    days = set(days)
    df = pd.DataFrame(columns=["DateTime", "MiddleLatitudeA", "MiddleLatitudeK", "HighLatitudeA", "HighLatitudeK", "EstimatedPlanetaryA", "EstimatedPlanetaryK"])
    for line in lines:
        ln = line.replace("-", " -").split()
        dt = " ".join(ln[:3])
        if dt not in days:
            continue
        MiddleLatitudeK = list()
        HighLatitudeK = list()
        EstimatedPlanetaryK = list()
        for i in range(8):
            df.loc[len(df)] = {"DateTime" : (dt + " " +str(i*3)), "MiddleLatitudeA": int(ln[3]), "MiddleLatitudeK": int(ln[4 + i]), "HighLatitudeA": int(ln[12]), "HighLatitudeK": int(ln[13 + i]), "EstimatedPlanetaryA": float(ln[21]), "EstimatedPlanetaryK": float(ln[22 + i])}
            # MiddleLatitudeK.append(int(ln[4 + i]))
            # HighLatitudeK.append(int(ln[13 + i]))
            # EstimatedPlanetaryK.append(int(ln[22 + i]))
        # Y_df.loc[len(Y_df)] = {"DateTime" : (dt + " " +str(i*3)), "MiddleLatitudeA": ln[3], "MiddleLatitudeK": MiddleLatitudeK, "HighLatitudeA": ln[12], "HighLatitudeK": HighLatitudeK, "EstimatedPlanetaryA": ln[21], "EstimatedPlanetaryK": MiddleLatitudeK}
    Y_df.append(df)

X_full = pd.concat(X_df)
y_full = pd.concat(Y_df)


X, y = X_full[[1, 2, 3]], y_full[['EstimatedPlanetaryK']]
input_seq_len = 28*12
output_seq_len = 28*3
x_temp = list()
y_temp = list()
for i in range(input_seq_len, int(len(y) / 8) - output_seq_len - 1):
    x_temp.append(X[1440 * (i - input_seq_len):1440 * (i)].values)
    y_temp.append(y[i * 8: 8 * (i + output_seq_len)].values)
print('a')
X_train = tf.cast(np.array(x_temp), tf.float32)
y_train = tf.cast(np.array(y_temp), tf.float32)

input_seq_length = 483840
input_features = 3
output_seq_length = 672
latent_dim = 64

# Define the encoder
encoder_inputs = layers.Input(shape=(input_seq_length, input_features))
encoder_lstm = layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Define the decoder
decoder_inputs = layers.RepeatVector(output_seq_length)(encoder_outputs)
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_outputs = layers.Dense(32, activation='relu')(decoder_outputs)
decoder_outputs = layers.Dense(16, activation='relu')(decoder_outputs)
decoder_outputs = layers.Dense(1, activation='relu')(decoder_outputs)

# Construct the model
model = tf.keras.Model(encoder_inputs, decoder_outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
print(model.summary())


model.fit(X_train, y_train, epochs=1000, batch_size=16)


model.save('models/model0')