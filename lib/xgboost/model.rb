module XGBoost
  class Model
    attr_reader :booster

    def initialize(n_estimators: 100, importance_type: "gain", early_stopping_rounds: nil, **options)
      @params = options
      @n_estimators = n_estimators
      @importance_type = importance_type
      @early_stopping_rounds = early_stopping_rounds
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
