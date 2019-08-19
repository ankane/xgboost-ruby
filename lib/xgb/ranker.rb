module Xgb
  class Ranker < Model
    def initialize(max_depth: 3, learning_rate: 0.1, n_estimators: 100, objective: "rank:pairwise", importance_type: "gain", **options)
      super
    end

    def fit(x, y, group)
      dtrain = DMatrix.new(x, label: y)
      dtrain.group = group
      @booster = Xgb.train(@params, dtrain, num_boost_round: @n_estimators)
      nil
    end
  end
end
