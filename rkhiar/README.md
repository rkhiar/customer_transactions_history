## Project structure :
The project is structured in two main parts :

### Analytics :  
rkhiar/scripts/queries.py

Answering the queries corresponding to the basic questions 1, 2 and 4.  
In order to demonstrate different data processing skills, the queries are written using (basic) Pandas DataFrames and SQL.  
The required visuals are produced using Matplotlib/seaborn.

### Predictive modeling  : 

#### Data set construction :
rkhiar/scripts/data_preprocess.py  

Row data is provided in a transaction list format. in order to perform the monthly sales predictive analysis, it has been aggregated based on this time dimension.
The logical approach of the data set construction is the following :  

To predict the number of transactions for the next three months  per customer at any time in 2019, the training will be based on a sequence of the last sliding year data. For instance : 

Prediction of 201901--201903 sales will be based on  201801--201812 data.  
Prediction of 201903--201906 sales will be based on  201803--201902 data ......  

Using Sql queries, data has been formatted this way.  

Because of this transaction format of the row data, the monthly aggregation will produce data only for months with sales. (For instance, if a customer doesn't have any transaction in 201804, we will not have data for this month/customer). These missing months have been added with a number of transactions equal to 0. The fact that a customer doesn't buy anything in a particular month is as important as if he does. 
This way, we'll produce a full twelve months sequence.  

Data is finally split in train and test.  


#### Model development : 
rkhiar/scripts/model.py  
rkhiar/scripts/evaluate.py  

The problem faced is a sequential regression one.
Since it is a univariate case, it can be solved using basic regression machine learning models or using a Recurrent Neural Network.

Model applied are : 
- Linear Regression.
- Support Vector Regression.
- Gradient Boosting Regression.
- Recurrent Neural Network.

These models were scored using Rsquared metric (appropriate one for regression case).
Gradient Boosting Regressor gives the best results with test R2 = 0.79. Model has been saved.  
An evaluation script loads the model and the test_data_set and makes predictions.

#### Model deployment : 
Deployment using a flask API receiving and returning json data :  
Inputs  :  **customer_ID, [list of 12 months nb_transaction]**  
Outputs  :  **customer_ID, the next 3 months nb_transaction prediction**  

Input example :  
{  
      "3915408" : [5,0,4,0,29,0,0,0,0,0,0,0],  
      "4485119" : [1,1,0,18,5,1,8,16,1,0,0,3],  
      "3551916" : [2,5,0,0,12,9,1,0,0,0,0,0]  
}  
  
The API is accessible using **POSTMAN with a JSON ["POST" or "GET"] request containing body as the input example described above**.  
Link : ec2-35-180-85-192.eu-west-3.compute.amazonaws.com:8080/forcast  

