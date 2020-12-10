import xgboost as xgb
import pandas as pd

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
eval_hist = xgb.cv(regression_params, regression_train, shuffle=False, as_pandas=False)
for k in ['train-rmse-mean', 'train-rmse-std', 'test-rmse-mean', 'test-rmse-std']:
  print(k, eval_hist[k][0])
  print(k, eval_hist[k][-1])

print()
print('test_binary')

binary_params = {'objective': 'binary:logistic'}
binary_train = xgb.DMatrix(X_train, label=y_train.replace(2, 1))
eval_hist = xgb.cv(binary_params, binary_train, shuffle=False, as_pandas=False)
for k in ['train-logloss-mean', 'train-logloss-std', 'test-logloss-mean', 'test-logloss-std']:
  print("%-20s %f" % (k, eval_hist[k][0]))
  print("%-20s %f" % (k, eval_hist[k][-1]))

print()
print('test_multiclass')

multiclass_params = {'objective': 'multi:softprob', 'num_class': 3}
multiclass_train = xgb.DMatrix(X_train, label=y_train)
eval_hist = xgb.cv(multiclass_params, multiclass_train, shuffle=False, as_pandas=False)
for k in ['train-mlogloss-mean', 'train-mlogloss-std', 'test-mlogloss-mean', 'test-mlogloss-std']:
  print("%-20s %f" % (k, eval_hist[k][0]))
  print("%-20s %f" % (k, eval_hist[k][-1]))

print('')
print('test_early_stopping_early')

eval_hist = xgb.cv(regression_params, regression_train, shuffle=False, as_pandas=False, verbose_eval=True, num_boost_round=100, early_stopping_rounds=5)
print(len(eval_hist['train-rmse-mean']))
