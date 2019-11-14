# Xgb

[XGBoost](https://github.com/dmlc/xgboost) - the high performance machine learning library - for Ruby

:fire: Uses the C API for blazing performance

[![Build Status](https://travis-ci.org/ankane/xgb.svg?branch=master)](https://travis-ci.org/ankane/xgb)

## Installation

First, [install XGBoost](https://xgboost.readthedocs.io/en/latest/build.html). For Homebrew, use:

```sh
brew install xgboost
```

Add this line to your application’s Gemfile:

```ruby
gem 'xgb'
```

## Getting Started

This library follows the [Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html), with the `get_` and `set_` prefixes removed from methods. Some methods and options are missing at the moment. PRs welcome!

## Learning API

Prep your data

```ruby
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [1, 2, 3, 4]
```

Train a model

```ruby
params = {objective: "reg:squarederror"}
dtrain = Xgb::DMatrix.new(x, label: y)
booster = Xgb.train(params, dtrain)
```

Predict

```ruby
dtest = Xgb::DMatrix.new(x)
booster.predict(dtest)
```

Save the model to a file

```ruby
booster.save_model("my.model")
```

Load the model from a file

```ruby
booster = Xgb::Booster.new(model_file: "my.model")
```

Get the importance of features

```ruby
booster.score
```

Early stopping

```ruby
Xgb.train(params, dtrain, evals: [[dtrain, "train"], [dtest, "eval"]], early_stopping_rounds: 5)
```

CV

```ruby
Xgb.cv(params, dtrain, nfold: 3, verbose_eval: true)
```

## Scikit-Learn API

Prep your data

```ruby
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [1, 2, 3, 4]
```

Train a model

```ruby
model = Xgb::Regressor.new
model.fit(x, y)
```

> For classification, use `Xgb::Classifier`

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
model.fit(x, y, eval_set: [[x_test, y_test]], early_stopping_rounds: 5)
```

## Data

Data can be an array of arrays

```ruby
[[1, 2, 3], [4, 5, 6]]
```

Or a Daru data frame

```ruby
Daru::DataFrame.from_csv("houses.csv")
```

Or a Numo NArray

```ruby
Numo::DFloat.new(3, 2).seq
```

## XGBoost Installation

There’s an experimental branch that includes XGBoost with the gem for easiest installation.

```ruby
gem 'xgb', github: 'ankane/xgb', branch: 'vendor', submodules: true
```

Please file an issue if it doesn’t work for you.

You can also specify the path to XGBoost in an initializer:

```ruby
Xgb.ffi_lib << "/path/to/xgboost/lib/libxgboost.so"
```

> Use `libxgboost.dylib` for Mac and `xgboost.dll` for Windows

## Helpful Resources

- [Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [Parameter Tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)

## Related Projects

- [LightGBM](https://github.com/ankane/lightgbm) - LightGBM for Ruby
- [Eps](https://github.com/ankane/eps) - Machine Learning for Ruby

## Credits

Thanks to the [xgboost](https://github.com/PairOnAir/xgboost-ruby) gem for serving as an initial reference.

## History

View the [changelog](https://github.com/ankane/xgb/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/xgb/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/xgb/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development and testing:

```sh
git clone https://github.com/ankane/xgb.git
cd xgb
bundle install
bundle exec rake test
```
