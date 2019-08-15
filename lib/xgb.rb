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
    def train(params, dtrain, num_boost_round: 10)
      booster = Booster.new(params: params)
      booster.set_param("num_feature", dtrain.num_col)

      num_boost_round.times do |iteration|
        booster.update(dtrain, iteration)
      end

      booster
    end
  end
end
