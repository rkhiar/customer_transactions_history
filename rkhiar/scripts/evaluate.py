#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:12:27 2020

@author: riad
"""

# General Modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# path definition 
PATH = '/home/riad/Devs_Python/datachef/customer_transactions_history/rkhiar'
DATA_PATH = os.path.join(PATH,'data')
WORK_PATH = os.path.join(PATH,'work')

# Load test data
df_test =  pd.read_csv(os.path.join(DATA_PATH,'test_data_set.csv'), delimiter = ',', index_col = 0)
X_test, y_test = df_test.drop(columns=['customer_id', 'output']), df_test['output']


# prediction
path_to_model = os.path.join(WORK_PATH,'model.sav')
def predict(model_path, X, y):
    ''' returns predicted transactiions based on historical input X
        returns model R2 score'''
    # load the model from disk + predictions
    loaded_model = pickle.load(open(path_to_model, 'rb'))
    predictions = loaded_model.predict(X).astype(int)
    r2 = loaded_model.score(X, y)
    return predictions, r2


model_outputs = predict(path_to_model, X_test, y_test)

print(f'model R2 score : {model_outputs[1]}')
print(f'model predictions : {model_outputs[0]}')