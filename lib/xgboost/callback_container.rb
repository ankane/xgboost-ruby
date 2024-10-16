module XGBoost
  class CallbackContainer
    attr_reader :aggregated_cv, :history

    def initialize(callbacks, is_cv: false)
      @callbacks = callbacks
      callbacks.each do |callback|
        unless callback.is_a?(TrainingCallback)
          raise TypeError, "callback must be an instance of XGBoost::TrainingCallback"
        end
      end

      @history = {}
      @is_cv = is_cv
    end

    def before_training(model)
      @callbacks.each do |callback|
        model = callback.before_training(model)
        if @is_cv
          unless model.is_a?(PackedBooster)
            raise TypeError, "before_training should return the model"
          end
        else
          unless model.is_a?(Booster)
            raise TypeError, "before_training should return the model"
          end
        end
      end
      model
    end

    def after_training(model)
      @callbacks.each do |callback|
        model = callback.after_training(model)
        if @is_cv
          unless model.is_a?(PackedBooster)
            raise TypeError, "after_training should return the model"
          end
        else
          unless model.is_a?(Booster)
            raise TypeError, "after_training should return the model"
          end
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
      if @is_cv
        scores = model.eval_set(epoch)
        scores = aggcv(scores)
        @aggregated_cv = scores
        update_history(scores, epoch)
      else
        evals ||= []
        evals.each do |_, name|
          if name.include?("-")
            raise ArgumentError, "Dataset name should not contain `-`"
          end
        end
        score = model.eval_set(evals, epoch)
        metric_score = parse_eval_str(score)
        update_history(metric_score, epoch)
      end

      @callbacks.any? do |callback|
        callback.after_iteration(model, epoch, @history)
      end
    end

    private

    def update_history(score, epoch)
      score.each do |d|
        name = d[0]
        s = d[1]
        if @is_cv
          std = d[2]
          x = [s, std]
        else
          x = s
        end
        splited_names = name.split("-")
        data_name = splited_names[0]
        metric_name = splited_names[1..].join("-")
        @history[data_name] ||= {}
        data_history = @history[data_name]
        data_history[metric_name] ||= []
        metric_history = data_history[metric_name]
        metric_history << x
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

    def aggcv(rlist)
      cvmap = {}
      idx = rlist[0].split[0]
      rlist.each do |line|
        arr = line.split
        arr[1..].each_with_index do |it, metric_idx|
          k, v = it.split(":")
          (cvmap[[metric_idx, k]] ||= []) << v.to_f
        end
      end
      msg = idx
      results = []
      cvmap.sort { |x| x[0][0] }.each do |(_, name), s|
        mean = mean(s)
        std = stdev(s)
        results << [name, mean, std]
      end
      results
    end

    def mean(arr)
      arr.sum / arr.size.to_f
    end

    # don't subtract one from arr.size
    def stdev(arr)
      m = mean(arr)
      sum = 0
      arr.each do |v|
        sum += (v - m) ** 2
      end
      Math.sqrt(sum / arr.size)
    end
  end
end
