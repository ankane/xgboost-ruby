require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = iris_data
    group = [20, 80]

    model = Xgb::Ranker.new
    model.fit(x_train, y_train, group)
    y_pred = model.predict(x_test)
    p y_pred[0, 6]
    expected = [3.690385, 4.999046, -3.8156319, 0.61984086, 0.30764353, 0.6986507]
    assert_elements_in_delta expected, y_pred[0, 6]

    model.save_model(tempfile)

    model = Xgb::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_feature_importances
    x_train, y_train, _, _ = iris_data
    group = [20, 80]

    model = Xgb::Ranker.new
    model.fit(x_train, y_train, group)

    p model.feature_importances
    expected = [0.04503533, 0.06504705, 0.55673695, 0.33318064]
    assert_elements_in_delta expected, model.feature_importances
  end
end
