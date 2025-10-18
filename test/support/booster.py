import xgboost as xgb
import pandas as pd
import json

df = pd.read_csv('test/support/data.csv')

X = df.drop(columns=['y'])
y = df['y']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

train_data = xgb.DMatrix(X_train, label=y_train)
bst = xgb.train({}, train_data)
bst.save_model('test/support/model.json')

bst = xgb.Booster(model_file='test/support/model.json')
print('score', bst.get_score())
bst.dump_model('/tmp/model.json', dump_format='json')

with open('/tmp/model.json') as f:
  booster_dump = json.load(f)[0]

print('split', booster_dump['split'])

print('attributes', bst.attributes())
