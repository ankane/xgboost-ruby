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
    assert_in_delta 0.036666666666666674, eval_hist["train-error-mean"].first
    assert_in_delta 0.006666666666666667, eval_hist["train-error-mean"].last
    assert_in_delta 0.008498365855987974, eval_hist["train-error-std"].first
    assert_in_delta 0.0062360956446232355, eval_hist["train-error-std"].last
    assert_in_delta 0.09999999999999999, eval_hist["test-error-mean"].first
    assert_in_delta 0.05666666666666667, eval_hist["test-error-mean"].last
    assert_in_delta 0.02160246899469287, eval_hist["test-error-std"].first
    assert_in_delta 0.012472191289246473, eval_hist["test-error-std"].last
  end

  def test_multiclass
    eval_hist = XGBoost.cv(multiclass_params, multiclass_train, shuffle: false)
    assert_in_delta 0.06, eval_hist["train-merror-mean"].first
    assert_in_delta 0.0016666666666666668, eval_hist["train-merror-mean"].last
    assert_in_delta 0.008164965809277263, eval_hist["train-merror-std"].first
    assert_in_delta 0.0023570226039551583, eval_hist["train-merror-std"].last
    assert_in_delta 0.15, eval_hist["test-merror-mean"].first
    assert_in_delta 0.11666666666666665, eval_hist["test-merror-mean"].last
    assert_in_delta 0.029439202887759492, eval_hist["test-merror-std"].first
    assert_in_delta 0.004714045207910314, eval_hist["test-merror-std"].last
  end

  def test_early_stopping_early
    eval_hist = XGBoost.cv(regression_params, regression_train, shuffle: false, num_boost_round: 100, early_stopping_rounds: 5)
    assert_equal 15, eval_hist["train-rmse-mean"].size
  end
end
