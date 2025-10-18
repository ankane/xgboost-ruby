require_relative "test_helper"

class CvTest < Minitest::Test
  def test_regression
    eval_hist = XGBoost.cv(regression_params, regression_train, shuffle: false)
    assert_in_delta 0.40130647066587777, eval_hist["train-rmse-mean"].first
    assert_in_delta 0.05131705086428223, eval_hist["train-rmse-mean"].last
    assert_in_delta 0.005782070203164552, eval_hist["train-rmse-std"].first
    assert_in_delta 0.002585536557357405, eval_hist["train-rmse-std"].last
    assert_in_delta 0.4549188381629605, eval_hist["test-rmse-mean"].first
    assert_in_delta 0.3262552684991104, eval_hist["test-rmse-mean"].last
    assert_in_delta 0.028011817754340963, eval_hist["test-rmse-std"].first
    assert_in_delta 0.009954421852476214, eval_hist["test-rmse-std"].last
  end

  def test_binary
    eval_hist = XGBoost.cv(binary_params, binary_train, shuffle: false)
    assert_in_delta 0.290831, eval_hist["train-logloss-mean"].first
    assert_in_delta 0.069389, eval_hist["train-logloss-mean"].last
    assert_in_delta 0.029254, eval_hist["train-logloss-std"].first
    assert_in_delta 0.004453, eval_hist["train-logloss-std"].last
    assert_in_delta 0.324324, eval_hist["test-logloss-mean"].first
    assert_in_delta 0.151608, eval_hist["test-logloss-mean"].last
    assert_in_delta 0.049756, eval_hist["test-logloss-std"].first
    assert_in_delta 0.039735, eval_hist["test-logloss-std"].last
  end

  def test_multiclass
    eval_hist = XGBoost.cv(multiclass_params, multiclass_train, shuffle: false)
    assert_in_delta 0.677055, eval_hist["train-mlogloss-mean"].first
    assert_in_delta 0.114060, eval_hist["train-mlogloss-mean"].last
    assert_in_delta 0.006876, eval_hist["train-mlogloss-std"].first
    assert_in_delta 0.007326, eval_hist["train-mlogloss-std"].last
    assert_in_delta 0.734124, eval_hist["test-mlogloss-mean"].first
    assert_in_delta 0.375596, eval_hist["test-mlogloss-mean"].last
    assert_in_delta 0.010682, eval_hist["test-mlogloss-std"].first
    assert_in_delta 0.029593, eval_hist["test-mlogloss-std"].last
  end

  def test_early_stopping_early
    eval_hist = XGBoost.cv(regression_params, regression_train, shuffle: false, num_boost_round: 100, early_stopping_rounds: 5)
    assert_equal 10, eval_hist["train-rmse-mean"].size
  end
end
