require "mkmf"

def run(command)
  puts ">> #{command}"
  unless system(command)
    raise "Command failed"
  end
end

dir = File.expand_path("../../vendor/xgboost", __dir__)

arch = RbConfig::CONFIG["arch"]
puts "Arch: #{arch}"
case arch
when /darwin/i
  Dir.chdir(dir) do
    # doesn't support:
    # - openmp (multicore)
    # - rabit (distributed)
    run "cp make/minimum.mk config.mk"
    run "make -j4"
  end
when /mingw/
  Dir.chdir(dir) do
    run "cp make/mingw64.mk config.mk"
    run "ridk exec make -j4"
  end
else
  Dir.chdir("#{dir}/build") do
    run "cmake .."
    run "make -j4"
  end
end

create_makefile("xgb")
