module XGBoost
  class TrainingCallback
    def before_training(model)
      # Run before training starts
      model
    end

    def after_training(model)
      # Run after training is finished
      model
    end

    def before_iteration(model, epoch, evals_log)
      # Run before each iteration. Returns true when training should stop.
      false
    end

    def after_iteration(model, epoch, evals_log)
      # Run after each iteration. Returns true when training should stop.
      false
    end
  end
end
