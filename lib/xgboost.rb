# dependencies
require "ffi"

# modules
require_relative "xgboost/utils"
require_relative "xgboost/booster"
require_relative "xgboost/callback_container"
require_relative "xgboost/cv_pack"
require_relative "xgboost/dmatrix"
require_relative "xgboost/packed_booster"
require_relative "xgboost/version"

# callbacks
require_relative "xgboost/training_callback"
require_relative "xgboost/early_stopping"
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
    def train(
      params,
      dtrain,
      num_boost_round: 10,
      evals: nil,
      maximize: nil,
      early_stopping_rounds: nil,
      evals_result: nil,
      verbose_eval: true,
      callbacks: nil
    )
      callbacks = callbacks.nil? ? [] : callbacks.dup
      evals ||= []

      bst = Booster.new(params: params, cache: [dtrain] + evals.map { |d| d[0] })

      if verbose_eval
        verbose_eval = verbose_eval == true ? 1 : verbose_eval
        callbacks << EvaluationMonitor.new(period: verbose_eval)
      end
      if early_stopping_rounds
        callbacks << EarlyStopping.new(rounds: early_stopping_rounds, maximize: maximize)
      end
      cb_container = CallbackContainer.new(callbacks)

      bst = cb_container.before_training(bst)

      num_boost_round.times do |i|
        break if cb_container.before_iteration(bst, i, dtrain, evals)
        bst.update(dtrain, i)
        break if cb_container.after_iteration(bst, i, dtrain, evals)
      end

      bst = cb_container.after_training(bst)

      if !evals_result.nil?
        evals_result.merge!(cb_container.history)
      end

      bst.reset
    end

    def cv(
      params,
      dtrain,
      num_boost_round: 10,
      nfold: 3,
      maximize: nil,
      early_stopping_rounds: nil,
      verbose_eval: nil,
      show_stdv: true,
      seed: 0,
      callbacks: nil,
      shuffle: true
    )
      results = {}
      cvfolds =
        mknfold(
          dall: dtrain,
          param: params,
          nfold: nfold,
          seed: seed,
          shuffle: shuffle
        )

      callbacks = callbacks.nil? ? [] : callbacks.dup

      if verbose_eval
        verbose_eval = verbose_eval == true ? 1 : verbose_eval
        callbacks << EvaluationMonitor.new(period: verbose_eval, show_stdv: show_stdv)
      end
      if early_stopping_rounds
        callbacks << EarlyStopping.new(rounds: early_stopping_rounds, maximize: maximize)
      end
      callbacks_container = CallbackContainer.new(callbacks, is_cv: true)

      booster = PackedBooster.new(cvfolds)
      callbacks_container.before_training(booster)

      num_boost_round.times do |i|
        break if callbacks_container.before_iteration(booster, i, dtrain, nil)
        booster.update(i)

        should_break = callbacks_container.after_iteration(booster, i, dtrain, nil)
        res = callbacks_container.aggregated_cv
        res.each do |key, mean, std|
          if !results.include?(key + "-mean")
            results[key + "-mean"] = []
          end
          if !results.include?(key + "-std")
            results[key + "-std"] = []
          end
          results[key + "-mean"] << mean
          results[key + "-std"] << std
        end

        if should_break
          results.keys.each do |k|
            results[k] = results[k][..booster.best_iteration]
          end
          break
        end
      end

      callbacks_container.after_training(booster)

      results
    end

    def lib_version
      major = ::FFI::MemoryPointer.new(:int)
      minor = ::FFI::MemoryPointer.new(:int)
      patch = ::FFI::MemoryPointer.new(:int)
      FFI.XGBoostVersion(major, minor, patch)
      "#{major.read_int}.#{minor.read_int}.#{patch.read_int}"
    end

    private

    def mknfold(dall:, param:, nfold:, seed:, shuffle:)
      rand_idx = (0...dall.num_row).to_a
      rand_idx.shuffle!(random: Random.new(seed)) if shuffle

      kstep = (rand_idx.size / nfold.to_f).ceil
      out_idset = rand_idx.each_slice(kstep).to_a[0...nfold]
      in_idset = []
      nfold.times do |i|
        idx = out_idset.dup
        idx.delete_at(i)
        in_idset << idx.flatten
      end

      ret = []
      nfold.times do |k|
        fold_dtrain = dall.slice(in_idset[k])
        fold_dvalid = dall.slice(out_idset[k])
        ret << CVPack.new(fold_dtrain, fold_dvalid, param)
      end
      ret
    end
  end
end
