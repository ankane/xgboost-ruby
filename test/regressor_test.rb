require_relative "test_helper"

class RegressorTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = regression_data

    model = XGBoost::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1.1791534423828125, 1.7054706811904907, 1.5178813934326172, 0.5577107071876526, 0.9730352163314819, 1.1123452186584473]
    assert_elements_in_delta expected, y_pred.first(6)

    expected = [0.10412569344043732, 0.3034818470478058, 0.47513794898986816, 0.11725451052188873]
    assert_elements_in_delta expected, model.feature_importances

    model.save_model(tempfile)

    model = XGBoost::Regressor.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = regression_data

    model = XGBoost::Regressor.new
    model.fit(x_train, y_train, early_stopping_rounds: 5, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 14, model.booster.best_iteration
  end

  def test_daru
    data = Daru::DataFrame.from_csv(data_path)
    y = data["y"]
    x = data.delete_vector("y")

    # daru has bug with 0...300
    x_train = x.row[0..299]
    y_train = y[0..299]
    x_test = x.row[300..-1]

    model = XGBoost::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1.1791534423828125, 1.7054706811904907, 1.5178813934326172, 0.5577107071876526, 0.9730352163314819, 1.1123452186584473]
    assert_elements_in_delta expected, y_pred.first(6)
  end
end
