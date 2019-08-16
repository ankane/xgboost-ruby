module Xgb
  class Classifier
    def initialize(max_depth: 3, learning_rate: 0.1, n_estimators: 100, objective: "binary:logistic", importance_type: "gain")
      @params = {
        max_depth: max_depth,
        objective: objective,
        learning_rate: learning_rate
      }
      @n_estimators = n_estimators
      @importance_type = importance_type
    end

    def fit(x, y)
      n_classes = y.uniq.size

      params = @params.dup
      if n_classes > 2
        params[:objective] = "multi:softprob"
        params[:num_class] = n_classes
      end

      dtrain = DMatrix.new(x, label: y)
      @booster = Xgb.train(params, dtrain, num_boost_round: @n_estimators)
      nil
    end

    def predict(data)
      dmat = DMatrix.new(data)
      y_pred = @booster.predict(dmat)

      if y_pred.first.is_a?(Array)
        # multiple classes
        y_pred.map do |v|
          v.map.with_index.max_by { |v2, i| v2 }.last
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
