# Predict Housing Price using Decision Tree Regressor Sklearn
# author: ikraduya

# imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# path to the data
file_path = 'data/train.csv'

train_data = pd.read_csv(file_path)

y = train_data.SalePrice  # extract the target predicted value
X = train_data.drop(['Id', 'SalePrice'], axis=1).select_dtypes(include=[np.number]) # select only numeric features

# split the data
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# imputation
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
train_X = imputer.fit_transform(train_X)
val_X = imputer.transform(val_X)

# create and fit model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

preds = forest_model.predict(val_X)

print('mae =', mean_absolute_error(val_y, preds))
