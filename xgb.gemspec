require_relative "lib/xgboost/version"

Gem::Specification.new do |spec|
  spec.name          = "xgb"
  spec.version       = XGBoost::VERSION
  spec.summary       = "High performance gradient boosting for Ruby"
  spec.homepage      = "https://github.com/ankane/xgboost-ruby"
  spec.license       = "Apache-2.0"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib,vendor}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 3"

  spec.add_dependency "ffi"
end
