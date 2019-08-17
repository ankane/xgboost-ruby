require "mkmf"

$dir = File.expand_path("../../vendor/xgboost/build", __dir__)

begin
  Dir.mkdir($dir)
rescue Errno::EEXIST
  # already exists
end

def run(command)
  puts ">> #{command}"
  # cd here to prevent warning
  unless system("cd #{$dir} && #{command}")
    raise "Command failed"
  end
end

arch = RbConfig::CONFIG["arch"]
if arch =~ /darwin/i
  # run "CC=gcc-8 CXX=g++-8 cmake .."
  run "make -j4"
else
  run "cmake .."
  run "make -j4"
end

create_makefile("xgb")
