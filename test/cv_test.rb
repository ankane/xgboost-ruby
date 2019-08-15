require_relative "test_helper"

class CvTest < Minitest::Test
  def test_regression
    eval_hist = Xgb.cv(regression_params, boston, shuffle: false, verbose_eval: true)
    assert_in_delta 17.023441, eval_hist["train-rmse-mean"].first
    assert_in_delta 1.518630, eval_hist["train-rmse-mean"].last
    assert_in_delta 1.578547, eval_hist["train-rmse-std"].first
    assert_in_delta 0.078385, eval_hist["train-rmse-std"].last
    assert_in_delta 16.903956, eval_hist["test-rmse-mean"].first
    assert_in_delta 5.233057, eval_hist["test-rmse-mean"].last
    assert_in_delta 3.611514, eval_hist["test-rmse-std"].first
    assert_in_delta 1.652534, eval_hist["test-rmse-std"].last
  end
end
