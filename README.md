# bike sharing data analysis
##  exploratory data analysis

### solution
1. import libraries and load dataset
2. check for missing values and outlier
3. features visualization

### results
1. find useful features for modelling
2. divide and explore categorical and numerical features
3. observe the basic situation of bike-sharing dataset through visualization


##  machine learning pipeline
### solution
1. load libraries and dataset
2. handle outlier value (windspeed)
3. add day type feature(weekday/weekend/holiday)
4. split dataset to training and testing parts
5. use 4 regressors to evaluate models
6. compare 4 regressors through MAE and RMSE

### results
| Model  | MAE score | RMSE score |
|:------:|:---------:|:----------:|
| linear |   103.65  |   139.52   |
| ridge  |   103.63  |   139.52   |
|  sgd   |   112.35  |   142.95   |
| lasso  |   103.29  |   139.67   |
those 4 models perform almost the same, Ridge Regression is slightly worse