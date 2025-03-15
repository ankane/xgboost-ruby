require_relative "test_helper"

class RegressorTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = regression_data

    model = XGBoost::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1.1507850885391235, 1.3788503408432007, 1.66087806224823, 0.5367319583892822, 1.0890085697174072, 1.3380072116851807]
    assert_elements_in_delta expected, y_pred.first(6)

    expected = [0.10230803489685059, 0.3298805356025696, 0.4885728359222412, 0.07923857867717743]
    assert_elements_in_delta expected, model.feature_importances

    model.save_model(tempfile)

    model = XGBoost::Regressor.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = regression_data

    model = XGBoost::Regressor.new(early_stopping_rounds: 5)
    model.fit(x_train, y_train, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 9, model.booster.best_iteration
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
    expected = [1.1507850885391235, 1.3788503408432007, 1.66087806224823, 0.5367319583892822, 1.0890085697174072, 1.3380072116851807]
    assert_elements_in_delta expected, y_pred.first(6)
  end
end
