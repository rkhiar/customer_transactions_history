#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:17:57 2020

@author: riad
"""


import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import pandasql as ps
from datetime import date
from dateutil.relativedelta import relativedelta
from dateutil import parser
import itertools


# path definition 
PATH = '/home/riad/Devs_Python/datachef/customer_transactions_history'
DATA_PATH = os.path.join(PATH,'data')
WORK_PATH = os.path.join(PATH,'work')

# Load files
df = pd.concat(map(lambda file: pd.read_csv(file, delimiter = ',', index_col = 0), \
                   glob.glob(os.path.join(DATA_PATH,'transactions*.csv'))))
# Change the date format to YYYYMM
df['date'] = df['date'].str[:7].str.replace('-', '')



#################################
### preprocessing data for model
#################################
"""
Add months without transactions in order to produce full years data
"""

# generate months
months_2017 = list(range(201701, 201713))
months_2018 = list(range(201801, 201813))
months_2019 = list(range(201901, 201913))
work_months = months_2017 + months_2018 + months_2019
work_months = list(map(str, work_months))

# list of customers ID
custumers=df.customer_id.unique().tolist()

# create a DF customer_id, months
mc = list(itertools.product(custumers, work_months))
mc_df = pd.DataFrame(mc, columns = ['customer_id','date'])

# aggregate the transactions
transactions = df.groupby(by = ['customer_id', 'date']).agg(
            nb_transaction = pd.NamedAgg(column = "product_id", aggfunc = "count")
            ).reset_index()

# full years data (months without any transaction are included)
full_transactions_df = transactions.merge(mc_df, on = ['customer_id', 'date'] , how = 'right').fillna(0)
 


# create the Data Set
base_date = parser.parse("2018-12-16T22:39:59.247Z")
data_set_list = []
for i in range (1,13):
    
    # built sliding period
    cursor_date = base_date + relativedelta(months = i-1)
    year_before = cursor_date + relativedelta(months = -11)
    tree_months_after =  cursor_date + relativedelta(months = 3)
    
    # format the dates to YYYYMM
    cursor_date = cursor_date.isoformat()[:7].replace('-', '')
    year_before = year_before.isoformat()[:7].replace('-', '')
    tree_months_after = tree_months_after.isoformat()[:7].replace('-', '')
    
    # get the sliding period data :      
    # data
    q1 = f""" SELECT customer_id, date, nb_transaction
        FROM full_transactions_df        
        where date between {year_before} and {cursor_date}
        order by customer_id, date asc
        """
    # outputs   
    q2 = f""" SELECT customer_id, sum(nb_transaction) as output
        FROM full_transactions_df        
        where date > {cursor_date} 
        and date <= {tree_months_after}
        group by customer_id
        """
                
    x = ps.sqldf(q1, locals())
    y = ps.sqldf(q2, locals())
           
    x = x.pivot(index = 'customer_id', columns = 'date', values = 'nb_transaction')
    x = x.rename(columns = dict(zip(list(x.columns), list(range(1,13)))))
    
    data_set = x.merge(y, on = 'customer_id', how = 'inner')           
    data_set_list.append(data_set)
    

data = pd.concat(data_set_list, ignore_index = False, axis = 0).astype(int).sample(frac = 1) 
data.to_csv(os.path.join(DATA_PATH,'data_set.csv'))

# splitting dataframe into train and test 
df_test, df_train = data[:4000], data[4000:]
df_train.to_csv(os.path.join(DATA_PATH,'train_data_set.csv'))
df_test.to_csv(os.path.join(DATA_PATH,'test_data_set.csv'))


