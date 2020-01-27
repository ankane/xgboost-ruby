require_relative "test_helper"

class CvTest < Minitest::Test
  def test_regression
    eval_hist = XGBoost.cv(regression_params, boston, shuffle: false)
    assert_in_delta 17.023441, eval_hist["train-rmse-mean"].first
    assert_in_delta 1.518630, eval_hist["train-rmse-mean"].last
    assert_in_delta 1.578547, eval_hist["train-rmse-std"].first
    assert_in_delta 0.078385, eval_hist["train-rmse-std"].last
    assert_in_delta 16.903956, eval_hist["test-rmse-mean"].first
    assert_in_delta 5.233057, eval_hist["test-rmse-mean"].last
    assert_in_delta 3.611514, eval_hist["test-rmse-std"].first
    assert_in_delta 1.652534, eval_hist["test-rmse-std"].last
  end

  def test_binary
    eval_hist = XGBoost.cv(binary_params, iris_binary, shuffle: false)
    assert_in_delta 0, eval_hist["train-error-mean"].first
    assert_in_delta 0, eval_hist["train-error-mean"].last
    assert_in_delta 0, eval_hist["train-error-std"].first
    assert_in_delta 0, eval_hist["train-error-std"].last
    assert_in_delta 0, eval_hist["test-error-mean"].first
    assert_in_delta 0, eval_hist["test-error-mean"].last
    assert_in_delta 0, eval_hist["test-error-std"].first
    assert_in_delta 0, eval_hist["test-error-std"].last
  end

  def test_multiclass
    eval_hist = XGBoost.cv(multiclass_params, iris, shuffle: false)
    assert_in_delta 0.013333, eval_hist["train-merror-mean"].first
    assert_in_delta 0.010000, eval_hist["train-merror-mean"].last
    assert_in_delta 0.009428, eval_hist["train-merror-std"].first
    assert_in_delta 0.008165, eval_hist["train-merror-std"].last
    assert_in_delta 0.06, eval_hist["test-merror-mean"].first
    assert_in_delta 0.06, eval_hist["test-merror-mean"].last
    assert_in_delta 0.043205, eval_hist["test-merror-std"].first
    assert_in_delta 0.043205, eval_hist["test-merror-std"].last
  end

  def test_early_stopping_early
    eval_hist = XGBoost.cv(regression_params, boston, shuffle: false, num_boost_round: 100, early_stopping_rounds: 5)
    assert_equal 25, eval_hist["train-rmse-mean"].size
  end
end
