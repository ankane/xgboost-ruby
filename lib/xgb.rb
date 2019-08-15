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
  end
end
