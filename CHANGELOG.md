## 0.4.0 (unreleased)

- Changed default `learning_rate` and `max_depth` for Scikit-Learn API to match Python
- Improved error message when OpenMP not found on Mac

## 0.3.1 (2020-04-16)

- Added `feature_names` and `feature_types` to `DMatrix`
- Added feature names to `dump`

## 0.3.0 (2020-02-19)

- Updated XGBoost to 1.0.0

## 0.2.1 (2020-02-11)

- Fixed `Could not find XGBoost` error on some Linux platforms
- Fixed `SignalException` on Windows

## 0.2.0 (2020-01-26)

- Prefer `XGBoost` over `Xgb`
- Changed to Apache 2.0 license to match XGBoost
- Added shared libraries
- Added support for booster attributes

## 0.1.3 (2019-10-27)

- Added support for missing values
- Fixed Daru training and prediction
- Fixed error with JRuby

## 0.1.2 (2019-08-19)

- Friendlier message when XGBoost not found
- Free memory when objects are destroyed
- Added `Ranker`
- Added early stopping to Scikit-Learn API

## 0.1.1 (2019-08-16)

- Added Scikit-Learn API
- Added early stopping
- Added `cv` method
- Added support for Daru and Numo::NArray
- Added many other methods
- Fixed shape of multiclass predictions when loaded from file

## 0.1.0 (2019-08-15)

- First release
