require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_train_regression
    y_test = boston_test.label

    model = Xgb.train(regression_params, boston_train) #, valid_sets: [boston_train, boston_test], verbose_eval: false)
    y_pred = model.predict(boston_test)
    assert_operator rsme(y_test, y_pred), :<=, 7

    model.save_model("/tmp/model.txt")
    model = Xgb::Booster.new(model_file: "/tmp/model.txt")
    y_pred = model.predict(boston_test)
    assert_operator rsme(y_test, y_pred), :<=, 7
  end

  def test_train_binary
    # map to binary
    iris_train = Xgb::DMatrix.new(iris_train().data, label: iris_train().label.map { |v| v > 1 ? 1.0 : v })
    iris_test = Xgb::DMatrix.new(iris_test().data, label: iris_test().label.map { |v| v > 1 ? 1.0 : v })

    model = Xgb.train(binary_params, iris_train) #, valid_sets: [iris_train, iris_test], verbose_eval: false)
    y_pred = model.predict(iris_test)
    assert_in_delta 0.96484315, y_pred[0]

    y_pred = model.predict(iris_test)
    assert_equal 50, y_pred.size

    model.save_model("/tmp/model.txt")
    model = Xgb::Booster.new(model_file: "/tmp/model.txt")
    y_pred2 = model.predict(iris_test)
    assert_equal y_pred, y_pred2
  end

  def test_train_multiclass
    model = Xgb.train(multiclass_params, iris_train) #, valid_sets: [iris_train, iris_test], verbose_eval: false)
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
    # TODO fix shape
    # y_pred2 = model.predict(iris_test)
    # assert_equal y_pred, y_pred2
  end

  def test_early_stopping_early
    model = Xgb.train(regression_params, boston_train, num_boost_round: 100, evals: [[boston_train, 'train'], [boston_test, 'eval']], early_stopping_rounds: 5, verbose_eval: false)
    assert_equal 8, model.best_iteration
  end

  def test_cv_regression
    eval_hist = Xgb.cv(regression_params, boston, shuffle: false, verbose_eval: true)
    p eval_hist
  end

  private

  def regression_params
    {objective: "reg:squarederror"}
  end

  def binary_params
    {objective: "binary:logistic"}
  end

  def multiclass_params
    {objective: "multi:softprob", num_class: 3}
  end

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
