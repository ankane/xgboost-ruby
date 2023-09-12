module XGBoost
  class Ranker < Model
    def initialize(objective: "rank:ndcg", **options)
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
