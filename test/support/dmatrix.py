import xgboost as xgb

data = [[1, 2], [3, 4]]
label = [1, 2]
dataset = xgb.DMatrix(data, label=label)
print(dataset.feature_names)
print(dataset.feature_types)
