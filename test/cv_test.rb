require_relative "test_helper"

class CvTest < Minitest::Test
  def test_regression
    eval_hist = XGBoost.cv(regression_params, regression_train, shuffle: false)
    assert_in_delta 0.5370283333333333, eval_hist["train-rmse-mean"].first
    assert_in_delta 0.05816966666666667, eval_hist["train-rmse-mean"].last
    assert_in_delta 0.01169096589489319, eval_hist["train-rmse-std"].first
    assert_in_delta 0.0051829699551083205, eval_hist["train-rmse-std"].last
    assert_in_delta 0.5703973333333333, eval_hist["test-rmse-mean"].first
    assert_in_delta 0.32477533333333336, eval_hist["test-rmse-mean"].last
    assert_in_delta 0.02923110268570487, eval_hist["test-rmse-std"].first
    assert_in_delta 0.010496456364993958, eval_hist["test-rmse-std"].last
  end

  def test_binary
    eval_hist = XGBoost.cv(binary_params, binary_train, shuffle: false)
    assert_in_delta 0.484217, eval_hist["train-logloss-mean"].first
    assert_in_delta 0.079635, eval_hist["train-logloss-mean"].last
    assert_in_delta 0.006151, eval_hist["train-logloss-std"].first
    assert_in_delta 0.005936, eval_hist["train-logloss-std"].last
    assert_in_delta 0.511421, eval_hist["test-logloss-mean"].first
    assert_in_delta 0.169691, eval_hist["test-logloss-mean"].last
    assert_in_delta 0.006912, eval_hist["test-logloss-std"].first
    assert_in_delta 0.030953, eval_hist["test-logloss-std"].last
  end

  def test_multiclass
    eval_hist = XGBoost.cv(multiclass_params, multiclass_train, shuffle: false)
    assert_in_delta 0.789279, eval_hist["train-mlogloss-mean"].first
    assert_in_delta 0.120741, eval_hist["train-mlogloss-mean"].last
    assert_in_delta 0.006837, eval_hist["train-mlogloss-std"].first
    assert_in_delta 0.009493, eval_hist["train-mlogloss-std"].last
    assert_in_delta 0.840854, eval_hist["test-mlogloss-mean"].first
    assert_in_delta 0.375845, eval_hist["test-mlogloss-mean"].last
    assert_in_delta 0.003656, eval_hist["test-mlogloss-std"].first
    assert_in_delta 0.027267, eval_hist["test-mlogloss-std"].last
  end

  def test_early_stopping_early
    eval_hist = XGBoost.cv(regression_params, regression_train, shuffle: false, num_boost_round: 100, early_stopping_rounds: 5)
    assert_equal 15, eval_hist["train-rmse-mean"].size
  end
end
