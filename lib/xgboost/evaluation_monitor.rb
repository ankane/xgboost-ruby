module XGBoost
  class EvaluationMonitor < TrainingCallback
    def initialize(period:, show_stdv: false)
      @show_stdv = show_stdv
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
          if log[-1].is_a?(Array)
            score = log[-1][0]
            stdv = log[-1][1]
          else
            score = log[-1]
          end
          msg += fmt_metric(data, metric_name, score, stdv)
        end
      end
      msg += "\n"
      puts msg

      false
    end

    private

    def fmt_metric(data, metric, score, std)
      if !std.nil? && @show_stdv
        "\t#{data + "-" + metric}:#{score}+#{std}"
      else
        "\t#{data + "-" + metric}:#{score}"
      end
    end
  end
end
