import xgboost as xgb
import pandas as pd
import numpy as np

df = pd.read_csv('test/support/data.csv')

X = df.drop(columns=['y'])
y = df['y']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

print('test_regression')

regression_params = {'objective': 'reg:squarederror'}
regression_train = xgb.DMatrix(X_train, label=y_train)
regression_test = xgb.DMatrix(X_test, label=y_test)
bst = xgb.train(regression_params, regression_train)
y_pred = bst.predict(regression_test)
print(np.sqrt(np.mean((y_pred - y_test)**2)))

print('')
print('test_binary')

binary_params = {'objective': 'binary:logistic'}
binary_train = xgb.DMatrix(X_train, label=y_train.replace(2, 1))
binary_test = xgb.DMatrix(X_test, label=y_test.replace(2, 1))
bst = xgb.train(binary_params, binary_train)
y_pred = bst.predict(binary_test)
print(y_pred[0])

print('')
print('test_multiclass')

multiclass_params = {'objective': 'multi:softprob', 'num_class': 3}
multiclass_train = xgb.DMatrix(X_train, label=y_train)
multiclass_test = xgb.DMatrix(X_test, label=y_test)
bst = xgb.train(multiclass_params, multiclass_train)
y_pred = bst.predict(multiclass_test)
print(y_pred[0].tolist())

print('')
print('test_early_stopping_early')

bst = xgb.train(regression_params, regression_train, num_boost_round=100, evals=[(regression_train, 'train'), (regression_test, 'test')], early_stopping_rounds=5)
print(bst.best_iteration)
