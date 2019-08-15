#!/usr/bin/env bash

set -e

CACHE_DIR=$HOME/xgboost/$XGBOOST_VERSION

if [ ! -d "$CACHE_DIR" ]; then
  git clone --recursive --branch v$XGBOOST_VERSION https://github.com/dmlc/xgboost
  mv xgboost $CACHE_DIR
  cd $CACHE_DIR
  mkdir build
  cd build
  cmake ..
  make -j4
else
  echo "XGBoost cached"
fi
