#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  12 10:27:27 2020

@author: riad
"""


# General Modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Deep Learning Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, LSTM, Dense, Input, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import seaborn as sns
sns.set()


# path definition 
PATH = '/home/riad/Devs_Python/datachef/customer_transactions_history/rkhiar'
DATA_PATH = os.path.join(PATH, 'data')
WORK_PATH = os.path.join(PATH, 'work')


"""
# Data set creation
df =  pd.read_csv(os.path.join(DATA_PATH, 'data_set.csv'), delimiter = ',', index_col = 0)

x = df.drop(columns = ['customer_id', 'output'])
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
"""


# load data
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train_data_set.csv'), delimiter = ',', index_col = 0)
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test_data_set.csv'), delimiter = ',', index_col = 0)

X_train, y_train = df_train.drop(columns = ['customer_id', 'output']), df_train['output']
X_test, y_test = df_test.drop(columns = ['customer_id', 'output']), df_test['output']


#########################
# Linear Regression
#########################
reg = LinearRegression()
reg.fit(X_train, y_train)
print(f'Linear Regression train R2 score : {reg.score(X_train, y_train)}')
print(f'Linear Regression test R2 score : {reg.score(X_test, y_test)}')



##########################
# SVR Regression  
##########################
svr = SVR(kernel = 'poly', degree = 1)
svr.fit(X_train, y_train)
print(f'SVR Regression train R2 score : {svr.score(X_train, y_train)}')
print(f'SVR Regression test R2 score : {svr.score(X_test, y_test)}')

# removing negative values
# trans_pred = svr.predict(X_test).clip(0, max(svr.predict(X_test)))
# print(r2_score(y_test, trans_pred))



##############################
# Gradient Boosting Regressor
#############################
params = {'n_estimators' : 550,
          'max_depth' : 2,         
          'learning_rate' : 0.01,
          'min_samples_leaf' : 15,          
          'loss' : 'ls'}

gbr = GradientBoostingRegressor(**params)
gbr.fit(X_train, y_train)

print(f'Gradient Boosting train R2 score : {gbr.score(X_train, y_train)}')
print(f'Gradient Boosting test R2 score : {gbr.score(X_test, y_test)}')
      
# save the model to disk
'''filename = os.path.join(WORK_PATH,'model.sav')
pickle.dump(gbr, open(filename, 'wb'))'''
 


#######################
# RNN
#######################
def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.math.divide(residual, total))
  return r2

# NN 
inputs = Input(shape = (12, 1))
bi_lstm = LSTM(30, activation = 'relu', return_sequences = True)(inputs)
bi_lstm = LSTM(30, activation = 'relu', return_sequences = False)(bi_lstm)
outputs = Dense(1)(bi_lstm)
model = Model(inputs = inputs, outputs = outputs)

model.compile(loss = tf.keras.losses.MSE, 
                           optimizer = tf.keras.optimizers.Adam(lr = 0.00001),
                           metrics = R_squared
                           )
 
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 50)
mc = ModelCheckpoint(os.path.join(WORK_PATH, 'rnn_model.h5'), monitor = 'val_loss', mode = 'min', save_best_only = False)
    
model.fit(X_train, y_train, epochs = 50, batch_size = 128, verbose = 1,
                  shuffle = False ,validation_data = (X_test, y_test)
                  ,callbacks = [es, mc]
                  )  

plt.figure()
plt.plot(model.history.history['loss'], label = 'train')
plt.plot(model.history.history['val_loss'], label = 'valid')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
             
plt.figure()
plt.plot(model.history.history['R_squared'], label = 'train')
plt.plot(model.history.history['val_R_squared'], label = 'valid')
plt.xlabel("epoch")
plt.ylabel("R_squared")
plt.legend()
    

####################
# Visuals
####################

y_test_predict = gbr.predict(X_test).reshape(y_test.shape[0], 1)
y_test_actual = np.array(y_test).reshape(y_test.shape[0], 1)

scaler = StandardScaler()
y_test_std = scaler.fit_transform(y_test_actual)
y_test_predict_std = scaler.fit_transform(y_test_predict)

# Plot actual nb_transaction vs predictive ones
plt.plot(list(range(0,y_test.shape[0])), y_test_std, color = 'green', label = 'acual') 
plt.plot(list(range(0,y_test.shape[0])), y_test_predict_std, alpha=0.4, \
         color = 'red', label = 'predicted') 
plt.xlabel('rows')
plt.ylabel('nb_transactions')
plt.title('True nb transactions VS predicted')
plt.legend()
plt.show()

# Scatter actual nb_transaction vs predictive ones
sns.scatterplot(y_test_std.reshape(-1), y_test_predict_std.reshape(-1))
plt.xlabel("True")
plt.ylabel("Predicted")
