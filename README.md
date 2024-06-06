# lstm

In this project, I attempted to forecast conditionally heteroskedastic time series data using a LSTM recurrent neural network.

## Overview
The data was first collected, cleaned, and normalized. Then, it was split into the *x.train*, *y.train*, *x.test*, and *y.test* arrays. 
The LSTM model was created and fit to the training data with the help of keras & tensorflow packages; evaluation was on the *x.test* array. 
For the evaluation, 3 different metrics of prediction accuracy are considered - RMSE, MAPE, and Pearson Correlation coefficient.
For further details on implementation, please see *lstm.R*


