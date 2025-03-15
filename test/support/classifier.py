import xgboost as xgb
import pandas as pd

df = pd.read_csv('test/support/data.csv')

X = df.drop(columns=['y'])
yb = df['y'].replace(2, 1)
ym = df['y']

X_train = X[:300]
yb_train = yb[:300]
ym_train = ym[:300]
X_test = X[300:]
yb_test = yb[300:]
ym_test = ym[300:]

print('test_binary')

model = xgb.XGBClassifier()
model.fit(X_train, yb_train)
print(model.predict(X_test)[0:100].tolist())
print(model.predict_proba(X_test)[0].tolist())
print(model.feature_importances_.tolist())

print()
print('test_multiclass')

model = xgb.XGBClassifier()
model.fit(X_train, ym_train)
print(model.predict(X_test)[0:100].tolist())
print(model.predict_proba(X_test)[0].tolist())
print(model.feature_importances_.tolist())

print()
print('test_early_stopping')
model = xgb.XGBClassifier(early_stopping_rounds=5)
model.fit(X_train, ym_train, eval_set=[(X_test, ym_test)])
print(model.get_booster().best_iteration)

print()
print('test_missing')

X_train_miss = X_train.copy()
X_test_miss = X_test.copy()
X_train_miss[X_train_miss == 3.7] = None
X_test_miss[X_test_miss == 3.7] = None
model = xgb.XGBClassifier()
model.fit(X_train_miss, ym_train)
print(model.predict(X_test_miss)[0:100].tolist())
print(model.feature_importances_.tolist())
