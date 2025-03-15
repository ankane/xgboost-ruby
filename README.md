# XGBoost Ruby

[XGBoost](https://github.com/dmlc/xgboost) - high performance gradient boosting - for Ruby

[![Build Status](https://github.com/ankane/xgboost-ruby/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/xgboost-ruby/actions)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem "xgb"
```

On Mac, also install OpenMP:

```sh
brew install libomp
```

## Learning API

Prep your data

```ruby
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [1, 2, 3, 4]
```

Train a model

```ruby
params = {objective: "reg:squarederror"}
dtrain = XGBoost::DMatrix.new(x, label: y)
booster = XGBoost.train(params, dtrain)
```

Predict

```ruby
dtest = XGBoost::DMatrix.new(x)
booster.predict(dtest)
```

Save the model to a file

```ruby
booster.save_model("my.model")
```

Load the model from a file

```ruby
booster = XGBoost::Booster.new(model_file: "my.model")
```

Get the importance of features

```ruby
booster.score
```

Early stopping

```ruby
XGBoost.train(params, dtrain, evals: [[dtrain, "train"], [dtest, "eval"]], early_stopping_rounds: 5)
```

CV

```ruby
XGBoost.cv(params, dtrain, nfold: 3, verbose_eval: true)
```

Set metadata about a model

```ruby
booster["key"] = "value"
booster.attributes
```

## Scikit-Learn API

Prep your data

```ruby
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [1, 2, 3, 4]
```

Train a model

```ruby
model = XGBoost::Regressor.new
model.fit(x, y)
```

> For classification, use `XGBoost::Classifier`

Predict

```ruby
model.predict(x)
```

> For classification, use `predict_proba` for probabilities

Save the model to a file

```ruby
model.save_model("my.model")
```

Load the model from a file

```ruby
model.load_model("my.model")
```

Get the importance of features

```ruby
model.feature_importances
```

Early stopping

```ruby
model = XGBoost::Regressor.new(early_stopping_rounds: 5)
model.fit(x, y, eval_set: [[x_test, y_test]])
```

## Data

Data can be an array of arrays

```ruby
[[1, 2, 3], [4, 5, 6]]
```

Or a Numo array

```ruby
Numo::NArray.cast([[1, 2, 3], [4, 5, 6]])
```

Or a Rover data frame

```ruby
Rover.read_csv("houses.csv")
```

Or a Daru data frame

```ruby
Daru::DataFrame.from_csv("houses.csv")
```

## Helpful Resources

- [Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [Parameter Tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)

## Related Projects

- [LightGBM](https://github.com/ankane/lightgbm) - LightGBM for Ruby
- [Eps](https://github.com/ankane/eps) - Machine learning for Ruby

## Credits

This library follows the [Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html), with the `get_` and `set_` prefixes removed from methods to make it more Ruby-like.

Thanks to the [xgboost](https://github.com/PairOnAir/xgboost-ruby) gem for showing how to use FFI.

## History

View the [changelog](https://github.com/ankane/xgboost-ruby/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/xgboost-ruby/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/xgboost-ruby/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/xgboost-ruby.git
cd xgboost-ruby
bundle install
bundle exec rake vendor:all
bundle exec rake test
```
