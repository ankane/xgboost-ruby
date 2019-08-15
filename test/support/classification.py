import xgboost as xgb
import pandas as pd

df = pd.read_csv('test/support/iris.csv')

X = df.drop(columns=['Species'])
y = df['Species']
y = y.replace(2, 1)

X_train = X[:100]
y_train = y[:100]
X_test = X[100:]
y_test = y[100:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
data = xgb.DMatrix(X, label=y)

param = {'objective': 'binary:logistic'}
# param = {'objective': 'multi:softprob', 'num_class': 3}

# bst = xgb.train(param, dtrain) #, valid_sets=[train_data, test_data], early_stopping_rounds=5)
# print(bst.predict(dtest)[0])
# print(bst.predict(dtest).shape)

# bst.save_model('/tmp/model.txt')
# bst = xgb.Booster(model_file='/tmp/model.txt')
# print(bst.predict(dtest).shape)

eval_dict = xgb.cv(param, data, shuffle=False, verbose_eval=True)
print(eval_dict)

