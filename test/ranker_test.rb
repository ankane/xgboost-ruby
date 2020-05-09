require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = binary_data
    group = [100, 200]

    model = XGBoost::Ranker.new
    model.fit(x_train, y_train, group)
    y_pred = model.predict(x_test)
    expected = [6.443103313446045, 2.2402842044830322, 5.526688575744629, 0.1207202672958374, 4.204965591430664, 5.640408992767334]
    assert_elements_in_delta expected, y_pred.first(6)

    expected = [0.09895384311676025, 0.19357523322105408, 0.6621919274330139, 0.04527907073497772]
    assert_elements_in_delta expected, model.feature_importances

    model.save_model(tempfile)

    model = XGBoost::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end
end
