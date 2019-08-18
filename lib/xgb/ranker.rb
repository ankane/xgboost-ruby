module Xgb
  class Ranker
    def initialize(max_depth: 3, learning_rate: 0.1, n_estimators: 100, objective: "rank:pairwise", importance_type: "gain")
      @params = {
        max_depth: max_depth,
        objective: objective,
        learning_rate: learning_rate
      }
      @n_estimators = n_estimators
      @importance_type = importance_type
    end

    def fit(x, y, group)
      dtrain = DMatrix.new(x, label: y)
      dtrain.group = group
      @booster = Xgb.train(@params, dtrain, num_boost_round: @n_estimators)
      nil
    end

    def predict(data)
      dmat = DMatrix.new(data)
      @booster.predict(dmat)
    end

    def save_model(fname)
      @booster.save_model(fname)
    end

    def load_model(fname)
      @booster = Booster.new(params: @params, model_file: fname)
    end

    def feature_importances
      score = @booster.score(importance_type: @importance_type)
      scores = @booster.feature_names.map { |k| score[k] || 0.0 }
      total = scores.sum.to_f
      scores.map { |s| s / total }
    end
  end
end
