require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = iris_data
    group = [10] * 10

    model = Xgb::Ranker.new
    model.fit(x_train, y_train, group)
    y_pred = model.predict(x_test)
    p y_pred[0, 6]
    expected = [3.9751444, 4.8430905, -3.7419944, 1.2090977, 0.22059655, 1.8301414]
    assert_elements_in_delta expected, y_pred[0, 6]

    model.save_model(tempfile)

    model = Xgb::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_feature_importances
    x_train, y_train, _, _ = iris_data
    group = [10] * 10

    model = Xgb::Ranker.new
    model.fit(x_train, y_train, group)

    p model.feature_importances
    expected = [0.10086838, 0.06466652, 0.5948677, 0.23959741]
    assert_elements_in_delta expected, model.feature_importances
  end
end
