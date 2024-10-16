module XGBoost
  class EarlyStopping < TrainingCallback
    def initialize(
      rounds:,
      metric_name: nil,
      data_name: nil,
      maximize: nil,
      save_best: false,
      min_delta: 0.0
    )
      @data = data_name
      @metric_name = metric_name
      @rounds = rounds
      @save_best = save_best
      @maximize = maximize
      @stopping_history = {}
      @min_delta = min_delta
      if @min_delta < 0
        raise ArgumentError, "min_delta must be greater or equal to 0."
      end

      @current_rounds = 0
      @best_scores = {}
      @starting_round = 0
      super()
    end

    def before_training(model)
      @starting_round = model.num_boosted_rounds
      model
    end

    def after_iteration(model, epoch, evals_log)
      epoch += @starting_round
      msg = "Must have at least 1 validation dataset for early stopping."
      if evals_log.keys.length < 1
        raise ArgumentError, msg
      end

      # Get data name
      if @data
        data_name = @data
      else
        # Use the last one as default.
        data_name = evals_log.keys[-1]
      end
      if !evals_log.include?(data_name)
        raise ArgumentError, "No dataset named: #{data_name}"
      end

      if !data_name.is_a?(String)
        raise TypeError, "The name of the dataset should be a string. Got: #{data_name.class.name}"
      end
      data_log = evals_log[data_name]

      # Get metric name
      if @metric_name
        metric_name = @metric_name
      else
        # Use last metric by default.
        metric_name = data_log.keys[-1]
      end
      if !data_log.include?(metric_name)
        raise ArgumentError, "No metric named: #{metric_name}"
      end

      # The latest score
      score = data_log[metric_name][-1]
      update_rounds(
        score, data_name, metric_name, model, epoch
      )
    end

    def after_training(model)
      if !@save_best
        return model
      end

      best_iteration = model.best_iteration
      best_score = model.best_score
      # model = model[..(best_iteration + 1)]
      model.best_iteration = best_iteration
      model.best_score = best_score
      model
    end

    private

    def update_rounds(score, name, metric, model, epoch)
      get_s = lambda do |value|
        value.is_a?(Array) ? value[0] : value
      end

      maximize = lambda do |new_, best|
        get_s.(new_) - @min_delta > get_s.(best)
      end

      minimize = lambda do |new_, best|
        get_s.(best) - @min_delta > get_s.(new_)
      end

      improve_op = @maximize ? maximize : minimize

      if @stopping_history.empty?
        # First round
        @current_rounds = 0
        @stopping_history[name] = {}
        @stopping_history[name][metric] = [score]
        @best_scores[name] = {}
        @best_scores[name][metric] = [score]
        model.set_attr(best_score: score, best_iteration: epoch)
      elsif !improve_op.(score, @best_scores[name][metric][-1])
        # Not improved
        @stopping_history[name][metric] << score
        @current_rounds += 1
      else
        # Improved
        @stopping_history[name][metric] << score
        @best_scores[name][metric] << score
        record = @stopping_history[name][metric][-1]
        model.set_attr(best_score: record, best_iteration: epoch)
        @current_rounds = 0
      end

      if @current_rounds >= @rounds
        # Should stop
        return true
      end
      false
    end
  end
end
