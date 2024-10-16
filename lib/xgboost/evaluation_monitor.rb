module XGBoost
  class EvaluationMonitor < TrainingCallback
    def initialize(verbose_eval:, early_stopping_rounds:)
      @verbose_eval = verbose_eval
      @early_stopping_rounds = early_stopping_rounds

      @best_score = nil
      @best_iter = nil
      @best_message = nil
    end

    def after_iteration(model, epoch, evals_log)
      if evals_log.empty?
        return false
      end

      data_name = evals_log.keys[-1]
      data_log = evals_log[data_name]
      metric_name = data_log.keys[-1]

      if @early_stopping_rounds && epoch == 0
        puts "Will train until #{data_name}-#{metric_name} hasn't improved in #{@early_stopping_rounds.to_i} rounds." if @verbose_eval
      end

      msg = "[#{epoch}]"
      evals_log.each do |data, metric|
        metric.each do |metric_name, log|
          stdv = nil
          score = log[-1]
          msg += fmt_metric(data, metric_name, score, stdv)
        end
      end
      msg += "\n"

      puts msg if @verbose_eval
      score = data_log[metric_name][-1]

      # TODO handle larger better
      if @best_score.nil? || score < @best_score
        @best_score = score
        @best_iter = epoch
        @best_message = msg
      elsif @early_stopping_rounds && epoch - @best_iter >= @early_stopping_rounds
        model.best_iteration = @best_iter
        model.best_score = @best_score
        puts "Stopping. Best iteration:\n#{@best_message}" if @verbose_eval
        return true
      end

      false
    end

    private

    def fmt_metric(data, metric, score, std)
      "\t#{data + "-" + metric}:#{score}"
    end
  end
end
