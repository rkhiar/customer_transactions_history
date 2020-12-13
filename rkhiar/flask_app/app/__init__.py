#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:04:57 2020

@author: riad
"""

from flask import Flask


app = Flask(__name__)

from app import views
