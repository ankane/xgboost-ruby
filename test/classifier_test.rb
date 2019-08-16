require_relative "test_helper"

class ClassifierTest < Minitest::Test
  def test_binary
    x_train, y_train, x_test, _ = iris_data_binary

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    assert_equal expected, y_pred

    model.save_model("/tmp/my.model")

    model = Xgb::Classifier.new
    model.load_model("/tmp/my.model")
    assert_equal y_pred, model.predict(x_test)
  end

  def test_multiclass
    x_train, y_train, x_test, _ = iris_data

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [2, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 0, 2, 1, 1, 2, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1]
    assert_equal expected, y_pred

    model.save_model("/tmp/my.model")

    model = Xgb::Classifier.new
    model.load_model("/tmp/my.model")
    assert_equal y_pred, model.predict(x_test)
  end

  def test_predict_proba_binary
    x_train, y_train, x_test, _ = iris_data_binary

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict_proba(x_test)
    expected = [0.01680386, 0.98319614]
    assert_equal expected.size, y_pred[0].size
    expected.zip(y_pred[0]) do |exp, act|
      assert_in_delta exp, act
    end
  end

  def test_predict_proba_multiclass
    x_train, y_train, x_test, _ = iris_data

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict_proba(x_test)
    expected = [0.00768452, 0.03547496, 0.9568406]
    assert_equal expected.size, y_pred[0].size
    expected.zip(y_pred[0]) do |exp, act|
      assert_in_delta exp, act
    end
  end
end
