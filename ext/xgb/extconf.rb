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
if arch.end_with?("-darwin18")
  run 'CC=gcc-8 CXX=g++-8 cmake -DOpenMP_C_LIB_NAMES="omp" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib ..'
elsif arch =~ /darwin/i
  run "CC=gcc-8 CXX=g++-8 cmake .."
else
  run "cmake .."
end
run "make -j4"

puts "done"
