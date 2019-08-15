# dependencies
require "ffi"

# modules
require "xgb/utils"
require "xgb/booster"
require "xgb/dmatrix"
require "xgb/ffi"
require "xgb/version"

module Xgb
  class Error < StandardError; end

  class << self
    def train(params, dtrain, num_boost_round: 10, evals: nil, early_stopping_rounds: nil, verbose_eval: true)
      booster = Booster.new(params: params)
      booster.set_param("num_feature", dtrain.num_col)
      evals ||= []

      num_boost_round.times do |iteration|
        booster.update(dtrain, iteration)

        if evals.any?
          puts booster.eval_set(evals, iteration) if verbose_eval
        end
      end

      booster
    end
  end
end
