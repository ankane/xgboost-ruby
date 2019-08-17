require "mkmf"

dir = File.expand_path("../../vendor/xgboost", __dir__)

Dir.chdir(dir) do
  run "cp make/minimum.mk config.mk"
  run "make -j4"
end

create_makefile("xgb")
