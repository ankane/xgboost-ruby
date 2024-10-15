module XGBoost
  class TrainingCallback
    def before_training(model: nil)
      # Run before training starts
      model
    end

    def after_training(model: nil)
      # Run after training is finished
      model
    end

    def before_iteration(model: nil, epoch: nil)
      # Run before each iteration. Returns true when training should stop.
      false
    end

    def after_iteration(model: nil, epoch: nil, history: nil)
      # Run after each iteration. Returns true when training should stop.
      false
    end
  end
end
