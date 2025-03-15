require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_regression
    model = XGBoost.train(regression_params, regression_train)
    y_pred = model.predict(regression_test)
    assert_operator rsme(regression_test.label, y_pred), :<=, 0.32

    model.save_model(tempfile)
    model = XGBoost::Booster.new(model_file: tempfile)
    y_pred = model.predict(regression_test)
    assert_operator rsme(regression_test.label, y_pred), :<=, 0.32
  end

  def test_binary
    model = XGBoost.train(binary_params, binary_train)
    y_pred = model.predict(binary_test)
    assert_in_delta 0.9892826, y_pred.first
    assert_equal 200, y_pred.size

    model.save_model(tempfile)
    model = XGBoost::Booster.new(model_file: tempfile)
    y_pred2 = model.predict(binary_test)
    assert_equal y_pred, y_pred2
  end

  def test_multiclass
    model = XGBoost.train(multiclass_params, multiclass_train)

    y_pred = model.predict(multiclass_test)
    expected = [0.04140469804406166, 0.8922330141067505, 0.06636224687099457]
    assert_elements_in_delta expected, y_pred.first
    # ensure reshaped
    assert_equal 200, y_pred.size
    assert_equal 3, y_pred.first.size

    model.save_model(tempfile)
    model = XGBoost::Booster.new(model_file: tempfile)
    y_pred2 = model.predict(multiclass_test)
    assert_equal y_pred, y_pred2
  end

  def test_early_stopping_early
    model = XGBoost.train(regression_params, regression_train, num_boost_round: 100, evals: [[regression_train, "train"], [regression_test, "eval"]], early_stopping_rounds: 5, verbose_eval: false)
    assert_equal 9, model.best_iteration
  end

  def test_evals_result
    evals_result = {}
    XGBoost.train(regression_params, regression_train, evals: [[regression_train, "train"], [regression_test, "eval"]], evals_result: evals_result, verbose_eval: false)
    assert evals_result["train"]["rmse"]
  end

  def test_lib_version
    assert_match(/\A\d+\.\d+\.\d+\z/, XGBoost.lib_version)
  end

  def test_feature_names_and_types
    model = XGBoost.train(regression_params, regression_train)
    assert_equal 4.times.map { |i| "f#{i}" }, model.feature_names
    assert_nil model.feature_types
  end

  private

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
