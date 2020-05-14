require_relative "lib/xgboost/version"

Gem::Specification.new do |spec|
  spec.name          = "xgb"
  spec.version       = XGBoost::VERSION
  spec.summary       = "High performance gradient boosting for Ruby"
  spec.homepage      = "https://github.com/ankane/xgboost"
  spec.license       = "Apache-2.0"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@chartkick.com"

  spec.files         = Dir["*.{md,txt}", "{lib,vendor}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.4"

  spec.add_dependency "ffi"

  spec.add_development_dependency "bundler"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "minitest", ">= 5"
  spec.add_development_dependency "daru"
  spec.add_development_dependency "numo-narray"
  spec.add_development_dependency "rover-df"
end
