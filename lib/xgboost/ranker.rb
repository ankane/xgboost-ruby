module XGBoost
  class Ranker < Model
    def initialize(n_estimators: 100, objective: "rank:pairwise", importance_type: "gain", **options)
      super
    end

    def fit(x, y, group)
      dtrain = DMatrix.new(x, label: y)
      dtrain.group = group
      @booster = XGBoost.train(@params, dtrain, num_boost_round: @n_estimators)
      nil
    end
  end
end
