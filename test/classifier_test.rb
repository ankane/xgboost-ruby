require_relative "test_helper"

class ClassifierTest < Minitest::Test
  def test_binary
    x_train, y_train, x_test, _ = binary_data

    model = XGBoost::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    y_pred_proba = model.predict_proba(x_test)
    expected = [4.673004150390625e-05, 0.9999532699584961]
    assert_elements_in_delta expected, y_pred_proba.first

    expected = [0.13950465619564056, 0.25203850865364075, 0.5016216039657593, 0.1068352460861206]
    assert_elements_in_delta expected, model.feature_importances

    model.save_model(tempfile)

    model = XGBoost::Classifier.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_multiclass
    x_train, y_train, x_test, _ = multiclass_data

    model = XGBoost::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    y_pred_proba = model.predict_proba(x_test)
    expected = [0.001134952763095498, 0.9439229965209961, 0.054942041635513306]
    assert_elements_in_delta expected, y_pred_proba.first

    expected = [0.17008447647094727, 0.33299949765205383, 0.38294878602027893, 0.11396723240613937]
    assert_elements_in_delta expected, model.feature_importances

    model.save_model(tempfile)

    model = XGBoost::Classifier.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = multiclass_data

    model = XGBoost::Classifier.new(early_stopping_rounds: 5)
    model.fit(x_train, y_train, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 18, model.booster.best_iteration
  end

  def test_missing
    x_train, y_train, x_test, _ = multiclass_data

    [x_train, x_test].each do |xt|
      xt.each do |x|
        x.size.times do |i|
          x[i] = nil if x[i] == 3.7
        end
      end
    end

    model = XGBoost::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    expected = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    expected = [0.15985175967216492, 0.3488382399082184, 0.3853622376918793, 0.10594776272773743]
    assert_elements_in_delta expected, model.feature_importances
  end
end
