# dependencies
require "ffi"

# modules
require_relative "xgboost/utils"
require_relative "xgboost/booster"
require_relative "xgboost/callback_container"
require_relative "xgboost/dmatrix"
require_relative "xgboost/version"

# callbacks
require_relative "xgboost/training_callback"
require_relative "xgboost/evaluation_monitor"

# scikit-learn API
require_relative "xgboost/model"
require_relative "xgboost/classifier"
require_relative "xgboost/ranker"
require_relative "xgboost/regressor"

module XGBoost
  class Error < StandardError; end

  class << self
    attr_accessor :ffi_lib
  end
  lib_path =
    if Gem.win_platform?
      "x64-mingw/xgboost.dll"
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      if RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i
        "arm64-darwin/libxgboost.dylib"
      else
        "x86_64-darwin/libxgboost.dylib"
      end
    elsif RbConfig::CONFIG["host_os"] =~ /linux-musl/i
      "x86_64-linux-musl/libxgboost.so"
    else
      if RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i
        "aarch64-linux/libxgboost.so"
      else
        "x86_64-linux/libxgboost.so"
      end
    end
  vendor_lib = File.expand_path("../vendor/#{lib_path}", __dir__)
  self.ffi_lib = [vendor_lib]

  # friendlier error message
  autoload :FFI, "xgboost/ffi"

  class << self
    def train(params, dtrain, num_boost_round: 10, evals: nil, early_stopping_rounds: nil, verbose_eval: true, callbacks: nil)
      callbacks = callbacks.nil? ? [] : callbacks.dup
      evals ||= []

      bst = Booster.new(params: params)

      # TODO move
      num_feature = dtrain.num_col
      bst.set_param("num_feature", num_feature)
      bst.feature_names = dtrain.feature_names
      bst.feature_types = dtrain.feature_types

      # TODO split into separate callbacks in 0.9.0
      if verbose_eval || early_stopping_rounds
        callbacks << EvaluationMonitor.new(verbose_eval: verbose_eval, early_stopping_rounds: early_stopping_rounds)
      end
      cb_container = CallbackContainer.new(callbacks)

      bst = cb_container.before_training(bst)

      num_boost_round.times do |i|
        break if cb_container.before_iteration(bst, i, dtrain, evals)
        bst.update(dtrain, i)
        break if cb_container.after_iteration(bst, i, dtrain, evals)
      end

      bst = cb_container.after_training(bst)

      bst
    end

    def cv(params, dtrain, num_boost_round: 10, nfold: 3, seed: 0, shuffle: true, verbose_eval: nil, show_stdv: true, early_stopping_rounds: nil)
      rand_idx = (0...dtrain.num_row).to_a
      rand_idx.shuffle!(random: Random.new(seed)) if shuffle

      kstep = (rand_idx.size / nfold.to_f).ceil
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
        booster = Booster.new(params: params)
        booster.set_param("num_feature", dtrain.num_col)
        cvfolds << [booster, fold_dtrain, fold_dvalid]
      end

      eval_hist = {}

      if early_stopping_rounds
        best_score = nil
        best_iter = nil
      end

      num_boost_round.times do |iteration|
        scores = {}

        cvfolds.each do |(booster, fold_dtrain, fold_dvalid)|
          booster.update(fold_dtrain, iteration)
          message = booster.eval_set([[fold_dtrain, "train"], [fold_dvalid, "test"]], iteration)

          res = message.split.map { |x| x.split(":") }[1..-1].map { |k, v| [k, v.to_f] }
          res.each do |k, v|
            (scores[k] ||= []) << v
          end
        end

        message_parts = ["[#{iteration}]"]

        last_mean = nil
        means = {}
        scores.each do |eval_name, vals|
          mean = mean(vals)
          stdev = stdev(vals)

          (eval_hist["#{eval_name}-mean"] ||= []) << mean
          (eval_hist["#{eval_name}-std"] ||= []) << stdev

          means[eval_name] = mean
          last_mean = mean

          if show_stdv
            message_parts << "%s:%g+%g" % [eval_name, mean, stdev]
          else
            message_parts << "%s:%g" % [eval_name, mean]
          end
        end

        if early_stopping_rounds
          score = last_mean
          # TODO handle larger better
          if best_score.nil? || score < best_score
            best_score = score
            best_iter = iteration
          elsif iteration - best_iter >= early_stopping_rounds
            eval_hist.each_key do |k|
              eval_hist[k] = eval_hist[k][0..best_iter]
            end
            break
          end
        end

        # put at end to keep output consistent with Python
        puts message_parts.join("\t") if verbose_eval
      end

      eval_hist
    end

    def lib_version
      major = ::FFI::MemoryPointer.new(:int)
      minor = ::FFI::MemoryPointer.new(:int)
      patch = ::FFI::MemoryPointer.new(:int)
      FFI.XGBoostVersion(major, minor, patch)
      "#{major.read_int}.#{minor.read_int}.#{patch.read_int}"
    end

    private

    def mean(arr)
      arr.sum / arr.size.to_f
    end

    # don't subtract one from arr.size
    def stdev(arr)
      m = mean(arr)
      sum = 0
      arr.each do |v|
        sum += (v - m) ** 2
      end
      Math.sqrt(sum / arr.size)
    end
  end
end
