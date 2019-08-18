module Xgb
  class Regressor < Model
    def initialize(max_depth: 3, learning_rate: 0.1, n_estimators: 100, objective: "reg:squarederror", importance_type: "gain")
      @params = {
        max_depth: max_depth,
        objective: objective,
        learning_rate: learning_rate
      }
      @n_estimators = n_estimators
      @importance_type = importance_type
    end

    def fit(x, y)
      dtrain = DMatrix.new(x, label: y)
      @booster = Xgb.train(@params, dtrain, num_boost_round: @n_estimators)
      nil
    end
  end
end
