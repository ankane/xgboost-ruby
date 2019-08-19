import xgboost as xgb
import pandas as pd

df = pd.read_csv('test/support/iris.csv')

X = df.drop(columns=['Species'])
y = df['Species']
# y = y.replace(2, 1)

X_train = X[:100]
y_train = y[:100]
X_test = X[100:]
y_test = y[100:]

model = xgb.XGBClassifier()
# model.fit(X_train, y_train)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=True)
# print(model.predict_proba(X_test))
print(model.feature_importances_)
