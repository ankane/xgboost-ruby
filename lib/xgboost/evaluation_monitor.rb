module XGBoost
  class EvaluationMonitor < TrainingCallback
    def initialize(period:)
      @period = period
    end

    def after_iteration(model, epoch, evals_log)
      if evals_log.empty?
        return false
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
      puts msg

      false
    end

    private

    def fmt_metric(data, metric, score, std)
      "\t#{data + "-" + metric}:#{score}"
    end
  end
end
