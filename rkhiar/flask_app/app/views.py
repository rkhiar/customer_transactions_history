#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:14:03 2020

@author: riad
"""


# Flask Modules
from app import app
from flask import render_template, request, redirect, jsonify, make_response

# General Modules
import pandas as pd
import numpy as np
import pickle
import sklearn





def predict(X):
    ''' returns predicted transactiions based on historical input X '''
    # load the model from disk + predictions
    loaded_model = pickle.load(open('model.sav', 'rb'))
    predictions = loaded_model.predict(X)   
    return predictions
             

##########################
# Roots 
##########################
    
@app.route("/forcast", methods=["POST", "GET"])
def json():
    
    # check if a request contain json
    # if yes, covert it to a python dictionnary, make predictions and return json response        
    if request.is_json:
        req = request.get_json()               
        x = pd.DataFrame(req.values())
        dict_out = dict(zip(req.keys(), map(int,predict(x))))                
        res = make_response(jsonify(dict_out), 200)
        
        return res
    else:
        return make_response(jsonify({"message": "Request body must be JSON"}), 400)

    
    
