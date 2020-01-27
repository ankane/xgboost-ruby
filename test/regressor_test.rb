require_relative "test_helper"

class RegressorTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = boston_data

    model = XGBoost::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [28.509018, 25.23551, 24.38023, 32.31889, 33.371517, 27.57522]
    assert_elements_in_delta expected, y_pred[0, 6]

    model.save_model(tempfile)

    model = XGBoost::Regressor.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_feature_importances
    x_train, y_train, _, _ = boston_data

    model = XGBoost::Regressor.new
    model.fit(x_train, y_train)

    expected = [0.01210404, 0.00495621, 0.01828066, 0.0, 0.01790345, 0.68894494, 0.01395558, 0.01747261, 0.01420494, 0.03188109, 0.03816482, 0.00890863, 0.13322297]
    assert_elements_in_delta expected, model.feature_importances
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = boston_data

    model = XGBoost::Regressor.new
    model.fit(x_train, y_train, early_stopping_rounds: 5, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 30, model.booster.best_iteration
  end

  def test_daru
    data = Daru::DataFrame.from_csv("test/data/boston/boston.csv")
    y = data["medv"]
    x = data.delete_vector("medv")

    # daru has bug with 0...300
    x_train = x.row[0..299]
    y_train = y[0..299]
    x_test = x.row[300..-1]

    model = XGBoost::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [28.509018, 25.23551, 24.38023, 32.31889, 33.371517, 27.57522]
    assert_elements_in_delta expected, y_pred[0, 6]
  end
end
