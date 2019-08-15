import xgboost as xgb
import pandas as pd

df = pd.read_csv('test/support/boston.csv')

X = df.drop(columns=['medv'])
y = df['medv']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
data = xgb.DMatrix(X, label=y)

param = {}
# param['verbosity'] = -1
# param['metric'] = ['l1', 'l2', 'rmse']

bst = xgb.train(param, dtrain, evals=[(dtest, 'eval'), (dtrain, 'train')])
# print(bst.predict(X_test)[:1])

# eval_dict = lgb.cv(param, dataset, shuffle=False, stratified=False, verbose_eval=True, early_stopping_rounds=5)
# print(eval_dict)
