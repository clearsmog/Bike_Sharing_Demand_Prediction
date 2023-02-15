import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import prettytable as pt

"""
modelling
"""

# load dataframe
df = pd.read_pickle('../data/bike_sharing.pkl')

# split dataset to training and testing parts
df_y = df['count']
df_x = df.drop(['count'], axis=1)
X_train,X_test,Y_train,Y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

# add regressors

linear = LinearRegression()

ridge = Ridge()

sgd = SGDRegressor()

lasso = Lasso()

"""
model evaluation
"""
# model prediction
regressors_dict = {'linear': linear, 'ridge': ridge, 'sgd':sgd, 'lasso': lasso}

table = pt.PrettyTable(['MODEL', 'MAE score','RMSE score'])

for regressor_key in regressors_dict.keys():
    regressor = regressors_dict[regressor_key]
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)

    # Mean Absolute Error
    mae = round(mean_absolute_error(predictions, Y_test), 2)
    # Root Mean Square Error
    rmse = math.sqrt(mean_squared_error(predictions, Y_test))

    table.add_row([regressor_key, mae, rmse])

print(table)


"""
model inference
"""
# test model score
for regressor_key in regressors_dict.keys():
    regressor = regressors_dict[regressor_key]
    regressor_score = regressor.score(X_train, Y_train)
    print(regressor_key + ' score: ',  regressor_score)