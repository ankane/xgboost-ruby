import xgboost as xgb
import pandas as pd

df = pd.read_csv('test/support/boston.csv')

X = df.drop(columns=['medv'])
y = df['medv']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

model = xgb.XGBRegressor()
# model.fit(X_train, y_train)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=True)
# print(model.predict(X_test))
print(model.feature_importances_)
