module XGBoost
  class CallbackContainer
    def initialize(callbacks)
      @callbacks = callbacks
      callbacks.each do |callback|
        unless callback.is_a?(TrainingCallback)
          raise TypeError, "callback must be an instance of XGBoost::TrainingCallback"
        end
      end

      @history = {}
    end

    def before_training(model)
      @callbacks.each do |callback|
        model = callback.before_training(model)
        unless model.is_a?(Booster)
          raise TypeError, "before_training should return the model"
        end
      end
      model
    end

    def after_training(model)
      @callbacks.each do |callback|
        model = callback.after_training(model)
        unless model.is_a?(Booster)
          raise TypeError, "after_training should return the model"
        end
      end
      model
    end

    def before_iteration(model, epoch, dtrain, evals)
      @callbacks.any? do |callback|
        callback.before_iteration(model, epoch, @history)
      end
    end

    def after_iteration(model, epoch, dtrain, evals)
      evals ||= []
      evals.each do |_, name|
        if name.include?("-")
          raise ArgumentError, "Dataset name should not contain `-`"
        end
      end
      score = model.eval_set(evals, epoch)
      metric_score = parse_eval_str(score)
      update_history(metric_score, epoch)

      @callbacks.any? do |callback|
        callback.after_iteration(model, epoch, @history)
      end
    end

    private

    def update_history(score, epoch)
      score.each do |d|
        name = d[0]
        s = d[1]
        x = s
        splited_names = name.split("-")
        data_name = splited_names[0]
        metric_name = splited_names[1..].join("-")
        @history[data_name] ||= {}
        data_history = @history[data_name]
        data_history[metric_name] ||= []
        metric_history = data_history[metric_name]
        metric_history << x.to_f
      end
    end

    # TODO move
    def parse_eval_str(result)
      splited = result.split[1..]
      # split up `test-error:0.1234`
      metric_score_str = splited.map { |s| s.split(":") }
      # convert to float
      metric_score = metric_score_str.map { |n, s| [n, s.to_f] }
      metric_score
    end
  end
end
