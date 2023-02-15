import pandas as pd
from sklearn.ensemble import RandomForestRegressor

"""
data ingestion
"""

# load dataset
dataset = pd.read_csv('../data/bike_sharing.csv')

"""
data preprocessing
"""

# drop unused variables
dataset = dataset.drop(columns=['dteday', 'instant', 'casual', 'registered', 'atemp'])

# rename columns
dataset.rename(columns={'yr': 'year', 'mnth': 'month', 'hr': 'hour', 'weathersit': 'weather',
                        'hum': 'humidity', 'cnt': 'count'}, inplace=True)


"""
feature engineering
"""

# handle windspeed outlier value
windspeed0 = dataset[dataset['windspeed'] == 0]
windspeed1 = dataset[dataset['windspeed'] != 0]

columns = ['year', 'month', 'hour', 'season', 'weather', 'temp', 'humidity']
train_X = windspeed1[columns]
train_y = windspeed1['windspeed']
pred_X = windspeed0[columns]

model_forest = RandomForestRegressor(n_estimators=1000, random_state=42)

model_forest.fit(train_X, train_y)

windspeed0['windspeed'] = model_forest.predict(pred_X)

dataset = pd.concat([windspeed1, windspeed0])


# add day type feature
def get_day_type(col):
    if col['workingday'] == 1:
        return 1
    elif col['holiday'] == 1:
        return 2
    else:
        return 3


dataset.insert(0, 'day_type', dataset.apply(get_day_type, axis=1))


# save dataframe
dataset.to_pickle(path='../data/bike_sharing.pkl')
