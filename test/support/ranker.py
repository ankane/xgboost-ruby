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

group_train = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

model = xgb.XGBRanker()
model.fit(X_train, y_train, group_train)

print('predict')
print(model.predict(X_test))

print('feature_importances_')
print(model.feature_importances_)
