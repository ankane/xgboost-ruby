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
  $dir = File.expand_path("../../vendor/xgboost", __dir__)
  run "cp make/minimum.mk config.mk"
  run "make -j4"
else
  run "cmake .."
  run "make -j4"
end

create_makefile("xgb")
