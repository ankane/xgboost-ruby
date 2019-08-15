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

# bst = xgb.train(param, dtrain, num_boost_round=10) #, evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=5)
# print(bst.get_dump())
# print(bst.get_score())
# bst.save_model("test/support/boston.model")
# print(bst.best_iteration)
# print(bst.predict(X_test)[:1])

eval_dict = xgb.cv(param, data, shuffle=False, stratified=False, verbose_eval=True)
# print(eval_dict)
