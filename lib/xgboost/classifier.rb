module XGBoost
  class Classifier < Model
    def initialize(n_estimators: 100, objective: "binary:logistic", importance_type: "gain", **options)
      super
    end

    def fit(x, y, eval_set: nil, early_stopping_rounds: nil, verbose: true)
      n_classes = y.uniq.size

      params = @params.dup
      if n_classes > 2
        params[:objective] = "multi:softprob"
        params[:num_class] = n_classes
      end

      dtrain = DMatrix.new(x, label: y)
      evals = Array(eval_set).map.with_index { |v, i| [DMatrix.new(v[0], label: v[1]), "validation_#{i}"] }

      @booster = XGBoost.train(params, dtrain,
        num_boost_round: @n_estimators,
        early_stopping_rounds: early_stopping_rounds || @early_stopping_rounds,
        verbose_eval: verbose,
        evals: evals
      )
      nil
    end

    def predict(data)
      y_pred = super(data)

      if y_pred.first.is_a?(Array)
        # multiple classes
        y_pred.map do |v|
          v.map.with_index.max_by { |v2, _| v2 }.last
        end
      else
        y_pred.map { |v| v > 0.5 ? 1 : 0 }
      end
    end

    def predict_proba(data)
      dmat = DMatrix.new(data)
      y_pred = @booster.predict(dmat)

      if y_pred.first.is_a?(Array)
        # multiple classes
        y_pred
      else
        y_pred.map { |v| [1 - v, v] }
      end
    end
  end
end
