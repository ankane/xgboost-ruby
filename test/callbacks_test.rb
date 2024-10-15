require_relative 'test_helper'

class CallbacksTest < Minitest::Test
  class MockCallback < XGBoost::TrainingCallback
    attr_reader :before_training_count, :after_training_count, :before_iteration_count, :after_iteration_count,
                :before_training_args, :after_training_args, :before_iteration_args, :after_iteration_args, :history

    def initialize
      @before_training_count = 0
      @after_training_count = 0
      @before_iteration_count = 0
      @after_iteration_count = 0
      @before_training_args = []
      @after_training_args = []
      @before_iteration_args = []
      @after_iteration_args = []
      @history = {}
    end

    def before_training(model: nil)
      @before_training_count += 1
      model
    end

    def after_training(model: nil)
      @after_training_count += 1
      model
    end

    def before_iteration(model: nil, epoch: nil)
      @before_iteration_count += 1
      @before_iteration_args << { epoch: epoch }
      true
    end

    def after_iteration(model: nil, epoch: nil, history: nil)
      @after_iteration_count += 1
      @history = history
      true
    end
  end

  def test_callback_raises_when_not_training_callback
    num_boost_round = 10

    assert_raises(ArgumentError, /callback must subclass XGBoost::TrainingCallback/) do
      XGBoost.train(
        regression_params,
        regression_train,
        num_boost_round: num_boost_round,
        callbacks: ['not a callback'],
        evals: [[regression_train, 'train'], [regression_test, 'eval']]
      )
    end
  end

  def test_callback
    callback = MockCallback.new
    num_boost_round = 10

    XGBoost.train(
      regression_params,
      regression_train,
      num_boost_round: num_boost_round,
      callbacks: [callback],
      evals: [[regression_train, 'train'], [regression_test, 'eval']]
    )

    assert_equal 1, callback.before_training_count
    assert_equal 1, callback.after_training_count
    assert_equal num_boost_round, callback.before_iteration_count
    assert_equal num_boost_round, callback.after_iteration_count

    # Verify arguments
    train_rmse = callback.history['train']['rmse']
    assert_equal num_boost_round, train_rmse.size
    train_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end
    eval_rmse = callback.history['eval']['rmse']
    assert_equal num_boost_round, eval_rmse.size
    eval_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end

    epochs = callback.before_iteration_args.map { |e| e[:epoch] }
    assert_equal (0...num_boost_round).to_a, epochs
  end

  def test_callback_breaks_on_before_iteration
    callback = MockCallback.new
    def callback.before_iteration(model: nil, epoch: nil)
      @before_iteration_count += 1
      @before_iteration_args << { epoch: epoch }
      # If any callback returns false, break
      epoch.even?
    end
    num_boost_round = 10

    XGBoost.train(
      regression_params,
      regression_train,
      num_boost_round: num_boost_round,
      callbacks: [callback],
      evals: [[regression_train, 'train'], [regression_test, 'eval']]
    )

    assert_equal 1, callback.before_training_count
    assert_equal 1, callback.after_training_count
    assert_equal 2, callback.before_iteration_count
    assert_equal 1, callback.after_iteration_count

    # Verify arguments
    train_rmse = callback.history['train']['rmse']
    assert_equal 1, train_rmse.size
    train_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end
    eval_rmse = callback.history['eval']['rmse']
    assert_equal 1, eval_rmse.size
    eval_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end

    epochs = callback.before_iteration_args.map { |e| e[:epoch] }
    assert_equal (0...2).to_a, epochs
  end

  def test_callback_breaks_on_after_iteration
    callback = MockCallback.new
    def callback.after_iteration(model: nil, epoch: nil, history: nil)
      @after_iteration_count += 1
      @history = history
      epoch < 7
    end
    num_boost_round = 10

    XGBoost.train(
      regression_params,
      regression_train,
      num_boost_round: num_boost_round,
      callbacks: [callback],
      evals: [[regression_train, 'train'], [regression_test, 'eval']]
    )

    assert_equal 1, callback.before_training_count
    assert_equal 1, callback.after_training_count
    assert_equal 8, callback.before_iteration_count
    assert_equal 8, callback.after_iteration_count

    # Verify arguments
    train_rmse = callback.history['train']['rmse']
    assert_equal 8, train_rmse.size
    train_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end
    eval_rmse = callback.history['eval']['rmse']
    assert_equal 8, eval_rmse.size
    eval_rmse.each do |value|
      assert_in_delta 0.00, value, 1.0
    end

    epochs = callback.before_iteration_args.map { |e| e[:epoch] }
    assert_equal (0...8).to_a, epochs
  end

  def test_updates_model_before_training
    callback = MockCallback.new
    def callback.before_training(model: nil)
      model['device'] = 'cuda:0'
      model
    end

    num_boost_round = 10

    model = XGBoost.train(
      regression_params,
      regression_train,
      num_boost_round: num_boost_round,
      callbacks: [callback],
      evals: [[regression_train, 'train'], [regression_test, 'eval']]
    )

    assert_equal model['device'], 'cuda:0'
  end
end
