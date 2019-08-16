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

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
print(model.predict_proba(X_test))
