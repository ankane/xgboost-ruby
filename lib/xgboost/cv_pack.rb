require "forwardable"

module XGBoost
  class CVPack
    extend Forwardable

    def_delegators :@bst, :num_boosted_rounds, :best_iteration=, :best_score=

    attr_reader :bst

    def initialize(dtrain, dtest, param)
      @dtrain = dtrain
      @dtest = dtest
      @watchlist = [[dtrain, "train"], [dtest, "test"]]
      @bst = Booster.new(params: param, cache: [dtrain, dtest])
    end

    def update(iteration)
      @bst.update(@dtrain, iteration)
    end

    def eval_set(iteration)
      @bst.eval_set(@watchlist, iteration)
    end
  end
end
