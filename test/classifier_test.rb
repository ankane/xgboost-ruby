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
    expected = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    y_pred_proba = model.predict_proba(x_test)
    expected = [0.0020083063282072544, 0.9485360383987427, 0.04945564642548561]
    assert_elements_in_delta expected, y_pred_proba.first

    expected = [0.160569965839386, 0.33308327198028564, 0.3969796299934387, 0.10936711728572845]
    assert_elements_in_delta expected, model.feature_importances

    model.save_model(tempfile)

    model = XGBoost::Classifier.new
    model.load_model(tempfile)
    # TODO fix
    # assert_equal y_pred, model.predict(x_test)
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = multiclass_data

    model = XGBoost::Classifier.new(early_stopping_rounds: 5)
    model.fit(x_train, y_train, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 23, model.booster.best_iteration
  end

  def test_missing
    x_train, y_train, x_test, _ = multiclass_data

    x_train = x_train.map(&:dup)
    x_test = x_test.map(&:dup)
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
    expected = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    expected = [0.15650030970573425, 0.33717694878578186, 0.39813780784606934, 0.10818499326705933]
    assert_elements_in_delta expected, model.feature_importances
  end
end
