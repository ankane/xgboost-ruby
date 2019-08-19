module Xgb
  class Classifier < Model
    def initialize(max_depth: 3, learning_rate: 0.1, n_estimators: 100, objective: "binary:logistic", importance_type: "gain", **options)
      super
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
      y_pred = super(data)

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
  end
end
