# Xgb

[XGBoost](https://github.com/dmlc/xgboost) - the high performance machine learning library - for Ruby

:fire: Uses the C API for blazing performance

[![Build Status](https://travis-ci.org/ankane/xgb.svg?branch=master)](https://travis-ci.org/ankane/xgb)

## Installation

First, [install XGBoost](https://xgboost.readthedocs.io/en/latest/build.html). On Mac, copy `lib/libxgboost.dylib` to `/usr/local/lib`.

Add this line to your application’s Gemfile:

```ruby
gem 'xgb'
```

## Getting Started

Train a model

```ruby
params = {objective: "reg:squarederror"}
dtrain = Xgb::DMatrix.new(x_train, label: y_train)
booster = Xgb.train(params, dtrain)
```

Predict

```ruby
booster.predict(x_test)
```

Save the model to a file

```ruby
booster.save_model("my.model")
```

Load the model from a file

```ruby
booster = Xgb::Booster.new(model_file: "my.model")
```

Get the importance of features [master]

```ruby
booster.score
```

## Early Stopping [master]

```ruby
Xgb.train(params, dtrain, evals: [[dtrain, "train"], [dtest, "eval"]], early_stopping_rounds: 5)
```

## CV [master]

```ruby
Xgb.cv(params, dtrain, nfold: 3, verbose_eval: true)
```

## Scikit-Learn API [master]

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

## Reference

This library follows the [Core Data Structure and Learning API](https://xgboost.readthedocs.io/en/latest/python/python_api.html) of the Python library. Some methods and options are missing at the moment. PRs welcome!

## Helpful Resources

- [Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [Parameter Tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)

## Related Projects

- [LightGBM](https://github.com/ankane/lightgbm) - LightGBM for Ruby
- [Eps](https://github.com/ankane/eps) - Machine Learning for Ruby

## Credits

Thanks to the [xgboost](https://github.com/PairOnAir/xgboost-ruby) gem for serving as an initial reference, and Selva Prabhakaran for the [test datasets](https://github.com/selva86/datasets).

## History

View the [changelog](https://github.com/ankane/xgb/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/xgb/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/xgb/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features
