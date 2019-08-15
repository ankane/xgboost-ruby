require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_regression
    y_test = boston_test.label

    model = Xgb.train(regression_params, boston_train)
    y_pred = model.predict(boston_test)
    assert_operator rsme(y_test, y_pred), :<=, 7

    model.save_model("/tmp/model.txt")
    model = Xgb::Booster.new(model_file: "/tmp/model.txt")
    y_pred = model.predict(boston_test)
    assert_operator rsme(y_test, y_pred), :<=, 7
  end

  def test_binary
    model = Xgb.train(binary_params, iris_train_binary)
    y_pred = model.predict(iris_test_binary)
    assert_in_delta 0.96484315, y_pred[0]

    y_pred = model.predict(iris_test)
    assert_equal 50, y_pred.size

    model.save_model("/tmp/model.txt")
    model = Xgb::Booster.new(model_file: "/tmp/model.txt")
    y_pred2 = model.predict(iris_test)
    assert_equal y_pred, y_pred2
  end

  def test_multiclass
    model = Xgb.train(multiclass_params, iris_train)
    y_pred = model.predict(iris_test)[0]
    assert_in_delta 0.02350763, y_pred[0]
    assert_in_delta 0.04084724, y_pred[1]
    assert_in_delta 0.93564516, y_pred[2]

    y_pred = model.predict(iris_test)
    # ensure reshaped
    assert_equal 50, y_pred.size
    assert_equal 3, y_pred.first.size

    model.save_model("/tmp/model.txt")
    model = Xgb::Booster.new(model_file: "/tmp/model.txt")
    y_pred2 = model.predict(iris_test)
    assert_equal y_pred, y_pred2
  end

  def test_early_stopping_early
    model = Xgb.train(regression_params, boston_train, num_boost_round: 100, evals: [[boston_train, "train"], [boston_test, "eval"]], early_stopping_rounds: 5, verbose_eval: false)
    assert_equal 8, model.best_iteration
  end

  private

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
