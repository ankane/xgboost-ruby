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

      if early_stopping_rounds
        best_score = nil
        best_iter = nil
        best_message = nil
      end

      num_boost_round.times do |iteration|
        booster.update(dtrain, iteration)

        if evals.any?
          message = booster.eval_set(evals, iteration)
          res = message.split.map { |x| x.split(":") }[1..-1].map { |k, v| [k, v.to_f] }

          if early_stopping_rounds && iteration == 0
            metric = res[-1][0]
            puts "Will train until #{metric} hasn't improved in #{early_stopping_rounds.to_i} rounds." if verbose_eval
          end

          puts message if verbose_eval
          score = res[-1][1]

          # TODO handle larger better
          if best_score.nil? || score < best_score
            best_score = score
            best_iter = iteration
            best_message = message
          else
            booster.best_iteration = best_iter
            puts "Stopping. Best iteration:\n#{best_message}" if verbose_eval
            break
          end
        end
      end

      booster
    end

    def cv(params, dtrain, num_boost_round: 10, nfold: 3, seed: 0, shuffle: true, verbose_eval: nil)
      rand_idx = (0...dtrain.num_row).to_a
      rand_idx.shuffle!(random: Random.new(seed)) if shuffle

      kstep = rand_idx.size / nfold
      test_id = rand_idx.each_slice(kstep).to_a[0...nfold]
      train_id = []
      nfold.times do |i|
        idx = test_id.dup
        idx.delete_at(i)
        train_id << idx.flatten
      end

      folds = train_id.zip(test_id)
      cvfolds = []
      folds.each do |(train_idx, test_idx)|
        fold_dtrain = dtrain.slice(train_idx)
        fold_dvalid = dtrain.slice(test_idx)
        booster = Booster.new
        booster.set_param("num_feature", dtrain.num_col)
        cvfolds << [booster, fold_dtrain, fold_dvalid]
      end

      eval_hist = {}

      num_boost_round.times do |iteration|
        cvfolds.each do |(booster, fold_dtrain, fold_dvalid)|
          booster.update(fold_dtrain, iteration)
          message = booster.eval_set([[fold_dtrain, "train"], [fold_dvalid, "test"]], iteration)
          p message
        end
      end

      eval_hist
    end
  end
end
