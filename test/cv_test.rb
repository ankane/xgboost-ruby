require_relative "test_helper"

class CvTest < Minitest::Test
  def test_regression
    eval_hist = Xgb.cv(regression_params, boston, shuffle: false, verbose_eval: true)
  end
end
