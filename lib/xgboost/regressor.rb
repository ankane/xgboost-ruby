module XGBoost
  class Regressor < Model
    def initialize(n_estimators: 100, objective: "reg:squarederror", importance_type: "gain", **options)
      super
    end

    def fit(x, y, eval_set: nil, early_stopping_rounds: nil, verbose: true)
      dtrain = DMatrix.new(x, label: y)
      evals = Array(eval_set).map.with_index { |v, i| [DMatrix.new(v[0], label: v[1]), "validation_#{i}"] }

      @booster = XGBoost.train(@params, dtrain,
        num_boost_round: @n_estimators,
        early_stopping_rounds: early_stopping_rounds || @early_stopping_rounds,
        verbose_eval: verbose,
        evals: evals
      )
      nil
    end
  end
end
