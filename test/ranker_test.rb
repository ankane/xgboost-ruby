require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = iris_data_binary
    group = [20, 80]

    model = XGBoost::Ranker.new
    model.fit(x_train, y_train, group)
    y_pred = model.predict(x_test)
    expected = [3.144913, 3.144913, -2.144913, 3.144913, 3.144913, 3.144913]
    assert_elements_in_delta expected, y_pred[0, 6]

    model.save_model(tempfile)

    model = XGBoost::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_feature_importances
    x_train, y_train, _, _ = iris_data_binary
    group = [20, 80]

    model = XGBoost::Ranker.new
    model.fit(x_train, y_train, group)

    expected = [0, 0, 1, 0]
    assert_elements_in_delta expected, model.feature_importances
  end
end
