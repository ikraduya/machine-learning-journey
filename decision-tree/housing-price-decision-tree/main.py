# Predict Housing Price using Decision Tree Regressor Sklearn
# author: ikraduya

# imports
import pandas as pd
import numpy as np
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
X = train_data.drop(['Id', 'SalePrice'], axis=1).select_dtypes(include=[np.number]) # just select the numeric features

# split the data
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# drop column with missing value
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]
reduced_train_X = train_X.drop(cols_with_missing, axis=1)
reduced_val_X = val_X.drop(cols_with_missing, axis=1)

# handling missing value
from sklearn.impute import SimpleImputer

# imputation
imputer = SimpleImputer()
imputed_train_X = imputer.fit_transform(train_X)
imputed_val_X = imputer.transform(val_X)

def score_dataset(train_X, val_X, train_y, val_y):
       # comparing different tree size with mae
       candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
       scores = { leaf: get_mae(leaf, train_X, val_X, train_y, val_y) for leaf in candidate_max_leaf_nodes }
       best_tree_size = min(scores, key=scores.get)
       best_mae = scores[best_tree_size]
       print('best tree size:', best_tree_size)
       print('MAE =', best_mae)

print('\nDataset reduced:')
score_dataset(reduced_train_X, reduced_val_X, train_y, val_y)
print('\nDataset imputed:')
score_dataset(imputed_train_X, imputed_val_X, train_y, val_y)

# final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
# final_model.fit(X, y)


