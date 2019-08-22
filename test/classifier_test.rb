require_relative "test_helper"

class ClassifierTest < Minitest::Test
  def test_binary
    x_train, y_train, x_test, _ = iris_data_binary

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    assert_equal expected, y_pred

    model.save_model(tempfile)

    model = Xgb::Classifier.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_multiclass
    x_train, y_train, x_test, _ = iris_data

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [2, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 0, 2, 1, 1, 2, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1]
    assert_equal expected, y_pred

    model.save_model(tempfile)

    model = Xgb::Classifier.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_predict_proba_binary
    x_train, y_train, x_test, _ = iris_data_binary

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict_proba(x_test)
    expected = [0.01680386, 0.98319614]
    assert_elements_in_delta expected, y_pred[0]
  end

  def test_predict_proba_multiclass
    x_train, y_train, x_test, _ = iris_data

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict_proba(x_test)
    expected = [0.00768452, 0.03547496, 0.9568406]
    assert_elements_in_delta expected, y_pred[0]
  end

  def test_feature_importances_binary
    x_train, y_train, _, _ = iris_data_binary

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)

    expected = [0.0, 0.0, 1.0, 0.0]
    assert_elements_in_delta expected, model.feature_importances
  end

  def test_feature_importances_multiclass
    x_train, y_train, _, _ = iris_data

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)

    expected = [0.05196636, 0.3298079, 0.48029527, 0.1379305]
    assert_elements_in_delta expected, model.feature_importances
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = iris_data

    model = Xgb::Classifier.new
    model.fit(x_train, y_train, early_stopping_rounds: 5, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 0, model.booster.best_iteration
  end

  def test_missing
    x_train, y_train, x_test, _ = iris_data

    [x_train, x_test].each do |xt|
      xt.each do |x|
        x[1] = nil if x[1] == 2.8
      end
    end

    model = Xgb::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    expected = [2, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 0, 2, 1, 1, 2, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1]
    assert_equal expected, y_pred

    expected = [0.05196636, 0.3298079, 0.48029527, 0.1379305]
    assert_elements_in_delta expected, model.feature_importances
  end
end
