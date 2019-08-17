require "mkmf"

dir = File.expand_path("../../vendor/xgboost/build", __dir__)

begin
  Dir.mkdir(dir)
rescue Errno::EEXIST
  # already exists
end

def run(command)
  unless system(command)
    raise "Command failed"
  end
end

Dir.chdir(dir) do
  run "cmake .."
  run "make -j4"
end
