import xgboost as xgb
import pandas as pd

df = pd.read_csv('test/support/data.csv')

X = df.drop(columns=['y'])
y = df['y'].replace(2, 1)

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

group = [100, 200]

model = xgb.XGBRanker()
model.fit(X_train, y_train, group)
print(model.predict(X_test)[0:6].tolist())
print(model.feature_importances_.tolist())
