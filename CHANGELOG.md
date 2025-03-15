## 0.10.0 (2025-03-15)

- Updated XGBoost to 3.0.0

## 0.9.0 (2024-10-17)

- Updated XGBoost to 2.1.1
- Added support for callbacks
- Added `num_features` and `save_config` methods to `Booster`
- Added `num_nonmissing` and `data_split_mode` methods to `DMatrix`
- Dropped support for Ruby < 3.1

## 0.8.0 (2023-09-13)

- Updated XGBoost to 2.0.0
- Dropped support for Ruby < 3

## 0.7.3 (2023-07-24)

- Fixed error with `dup` and `clone`

## 0.7.2 (2023-05-12)

- Updated XGBoost to 1.7.5
- Added musl shared library for Linux
- Improved error message for invalid matrix

## 0.7.1 (2022-10-31)

- Updated XGBoost to 1.7.0

## 0.7.0 (2022-06-05)

- Updated XGBoost to 1.6.1
- Improved ARM detection
- Dropped support for Ruby < 2.7

## 0.6.0 (2021-10-23)

- Updated XGBoost to 1.5.0

## 0.5.3 (2021-05-12)

- Updated XGBoost to 1.4.0
- Added ARM shared library for Linux

## 0.5.2 (2021-03-09)

- Added ARM shared library for Mac

## 0.5.1 (2021-02-08)

- Fixed error with validation sets without early stopping

## 0.5.0 (2020-12-12)

- Updated XGBoost to 1.3.0

## 0.4.1 (2020-08-26)

- Updated XGBoost to 1.2.0

## 0.4.0 (2020-05-17)

- Updated XGBoost to 1.1.0
- Changed default `learning_rate` and `max_depth` for Scikit-Learn API to match Python
- Added support for Rover
- Improved performance of Numo datasets
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
