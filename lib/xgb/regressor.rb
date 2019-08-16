module Xgb
  class Regressor
    def initialize(max_depth: 3, learning_rate: 0.1, n_estimators: 100, objective: "reg:squarederror")
      @params = {
        max_depth: max_depth,
        objective: objective,
        learning_rate: learning_rate
      }
      @n_estimators = n_estimators
    end

    def fit(x, y)
      dtrain = DMatrix.new(x, label: y)
      @booster = Xgb.train(@params, dtrain, num_boost_round: @n_estimators)
      nil
    end

    def predict(data)
      dmat = DMatrix.new(data)
      @booster.predict(dmat)
    end
  end
end
