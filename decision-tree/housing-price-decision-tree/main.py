# Predict Housing Price using Decision Tree Regressor Sklearn
# author: ikraduya

# imports
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# functions
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
       model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
       model.fit(train_X, train_y)
       preds_val = model.predict(val_X)
       mae = mean_absolute_error(val_y, preds_val)
       return mae

# path to the data
file_path = 'data/train.csv'

train_data = pd.read_csv(file_path)

y = train_data.SalePrice  # extract the target predicted value

# choosing "Features"
feature_list = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']

selected_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[selected_features]

# split the data
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# create and fit model
housing_price_model = DecisionTreeRegressor(random_state=1)

# fit model
housing_price_model.fit(train_X, train_y)
val_predictions = housing_price_model.predict(val_X)

print('Prediction:', housing_price_model.predict(val_X.head()))
print('Actual values:', val_y.head().tolist())

# comparing different tree size with mae
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = { leaf: get_mae(leaf, train_X, val_X, train_y, val_y) for leaf in candidate_max_leaf_nodes }
best_tree_size = min(scores, key=scores.get)
print('best tree size', best_tree_size)

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)
