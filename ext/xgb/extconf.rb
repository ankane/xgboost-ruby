require "mkmf"

def run(command)
  puts ">> #{command}"
  unless system(command)
    raise "Command failed"
  end
end

dir = File.expand_path("../../vendor/xgboost", __dir__)

Dir.chdir(dir) do
  run "cp make/minimum.mk config.mk"
  run "make -j4"
end

create_makefile("xgb")
