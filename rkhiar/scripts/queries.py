#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:16:11 2020

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
import seaborn as sns

sns.set()



# path definition 
DATA_PATH = '/home/riad/Devs_Python/datachef/customer_transactions_history/data'

# Load file within my data path
df = pd.concat(map(lambda file: pd.read_csv(file, delimiter = ',', index_col=0), \
                   glob.glob(os.path.join(DATA_PATH,'transactions*.csv'))))
    
# Change the date format to YYYYMM
# This date/month format allows an easy handling as string or integer
df['date'] = df['date'].str[:7].str.replace('-', '')



#####################################################################################
# Create an ordered (descending) plot that shows the total number of transactions
# per customer from the most active customer to the least active one.
#####################################################################################

# query
query_00 = df.groupby('customer_id').agg(
            count_col = pd.NamedAgg(column = "product_id", aggfunc = "count")
            ).sort_values('count_col', ascending = False)

# limit records for visual matters
query_01 = query_00.head(100)

# plot
x = list(map(str, query_01.index.values))
y = query_01['count_col'].values.tolist()
plt.figure(figsize=(30,10))
plt.bar(x, y, align = 'center')
plt.xlabel('Customer ID')
plt.ylabel('Nb Transactions')
plt.title('Total number of transactions per customer')
#plt.ylim(0, 10000)
plt.show()

del query_00, query_01



#####################################################################################
# Given any product ID, create a plot to show its transaction frequency per month 
# for the year 2018
#####################################################################################

customer = 1001614
# aggregate the transactions
query_10 = df.groupby(by = ['customer_id', 'date']).agg(
            count_col = pd.NamedAgg(column = "product_id", aggfunc = "count")
            ).reset_index()

# Apply filters 
query_11 = query_10[(query_10['customer_id'] == customer) & (query_10['date'].str.contains("2019"))] \
           .sort_values('date', ascending=True)

# plot
x = list(map(str, query_11['date']))
y = query_11['count_col'].values.tolist()
plt.figure(figsize=(30,10))
plt.bar(x, y, align = 'center')
plt.xlabel('Months')
plt.ylabel('Nb Transactions')
plt.title('Transaction frequency per month/customer')
plt.show()

del query_10, query_11



#####################################################################################
# At any time, what are the top 5 products that drove the highest sales over the 
# last six months? 
#####################################################################################

# instantiate a date (date.today() or anytime date)
base_date = parser.parse("2019-03-16T22:39:59.247Z")
# add -6 months 
six_months_before = base_date + relativedelta(months=-6)
# format the dates to YYYYMM
six_months_before = six_months_before.isoformat()[:7].replace('-', '')
base_date = base_date.isoformat()[:7].replace('-', '')

# query :       
q1 = f""" SELECT product_id, count(*) 
        FROM df        
        where date between {six_months_before} and {base_date}
        group by product_id
        order by count(*) desc
        LIMIT 5;
      """
print(ps.sqldf(q1, locals()))



#####################################################################################
# Do you see a seasonality effect in this data set?
#####################################################################################

# query :       
q1 = f""" SELECT date, count(*) as nb_transaction
        FROM df  
        where date between 201701 and 201912
        group by date
        order by date asc
        """
query = ps.sqldf(q1, locals())

# Plot actual nb_transaction vs predictive ones
plt.figure(figsize=(30,10))
plt.plot(query['date'], query['nb_transaction'], color = 'green', label = 'acual') 
plt.xlabel('months')
plt.ylabel('nb_transaction')
plt.legend()
plt.show()

''' Visually, this full aggregated transaction per month shows a pattern that may obey
a seasonality behaviour.
For a more rigorous approach we can use statistical test on auto correlation coefficient '''