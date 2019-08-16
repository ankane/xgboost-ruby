require_relative "test_helper"

class ClassifierTest < Minitest::Test
  def test_binary
    x_train, y_train, x_test, _ = iris_data_binary

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    assert_equal expected, y_pred
  end

  def test_multiclass
    x_train, y_train, x_test, _ = iris_data

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [2, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 0, 2, 1, 1, 2, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1]
    assert_equal expected, y_pred
  end
end
