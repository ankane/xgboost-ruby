require_relative "test_helper"

class MockCallback < XGBoost::TrainingCallback
  attr_reader :before_training_count, :after_training_count, :before_iteration_count,
              :after_iteration_count, :before_iteration_args, :history

  def initialize
    @before_training_count = 0
    @after_training_count = 0
    @before_iteration_count = 0
    @after_iteration_count = 0
    @before_iteration_args = []
    @history = {}
  end

  def before_training(model)
    @before_training_count += 1
    model
  end

  def after_training(model)
    @after_training_count += 1
    model
  end

  def before_iteration(model, epoch, evals_log)
    @before_iteration_count += 1
    @before_iteration_args << {epoch: epoch}
    false
  end

  def after_iteration(model, epoch, evals_log)
    @after_iteration_count += 1
    @history = evals_log
    false
  end
end

class CallbacksTest < Minitest::Test
  def test_callback_raises_when_not_training_callback
    error = assert_raises(TypeError) do
      XGBoost.train(regression_params, regression_train, callbacks: [Object.new])
    end
    assert_equal "callback must be an instance of XGBoost::TrainingCallback", error.message
  end

  def test_callback
    callback = MockCallback.new
    num_boost_round = 10

    XGBoost.train(
      regression_params,
      regression_train,
      num_boost_round: num_boost_round,
      callbacks: [callback],
      evals: [[regression_train, "train"], [regression_test, "eval"]],
      verbose_eval: false
    )

    assert_equal 1, callback.before_training_count
    assert_equal 1, callback.after_training_count
    assert_equal num_boost_round, callback.before_iteration_count
    assert_equal num_boost_round, callback.after_iteration_count

    # Verify arguments
    train_rmse = callback.history["train"]["rmse"]
    assert_equal num_boost_round, train_rmse.size
    train_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end
    eval_rmse = callback.history["eval"]["rmse"]
    assert_equal num_boost_round, eval_rmse.size
    eval_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end

    epochs = callback.before_iteration_args.map { |e| e[:epoch] }
    assert_equal (0...num_boost_round).to_a, epochs
  end

  def test_callback_breaks_on_before_iteration
    callback = MockCallback.new
    def callback.before_iteration(model, epoch, evals_log)
      @before_iteration_count += 1
      @before_iteration_args << {epoch: epoch}
      epoch.odd?
    end

    XGBoost.train(
      regression_params,
      regression_train,
      callbacks: [callback],
      evals: [[regression_train, "train"], [regression_test, "eval"]],
      verbose_eval: false
    )

    assert_equal 1, callback.before_training_count
    assert_equal 1, callback.after_training_count
    assert_equal 2, callback.before_iteration_count
    assert_equal 1, callback.after_iteration_count

    # Verify arguments
    train_rmse = callback.history["train"]["rmse"]
    assert_equal 1, train_rmse.size
    train_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end
    eval_rmse = callback.history["eval"]["rmse"]
    assert_equal 1, eval_rmse.size
    eval_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end

    epochs = callback.before_iteration_args.map { |e| e[:epoch] }
    assert_equal (0...2).to_a, epochs
  end

  def test_callback_breaks_on_after_iteration
    callback = MockCallback.new
    def callback.after_iteration(model, epoch, evals_log)
      @after_iteration_count += 1
      @history = evals_log
      epoch >= 7
    end

    XGBoost.train(
      regression_params,
      regression_train,
      callbacks: [callback],
      evals: [[regression_train, "train"], [regression_test, "eval"]],
      verbose_eval: false
    )

    assert_equal 1, callback.before_training_count
    assert_equal 1, callback.after_training_count
    assert_equal 8, callback.before_iteration_count
    assert_equal 8, callback.after_iteration_count

    # Verify arguments
    train_rmse = callback.history["train"]["rmse"]
    assert_equal 8, train_rmse.size
    train_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end
    eval_rmse = callback.history["eval"]["rmse"]
    assert_equal 8, eval_rmse.size
    eval_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end

    epochs = callback.before_iteration_args.map { |e| e[:epoch] }
    assert_equal (0...8).to_a, epochs
  end

  def test_updates_model_before_training
    callback = MockCallback.new
    def callback.before_training(model)
      model["device"] = "cuda:0"
      model
    end

    model = XGBoost.train(regression_params, regression_train, callbacks: [callback])

    assert_equal model["device"], "cuda:0"
  end
end
