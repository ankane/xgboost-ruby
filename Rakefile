require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false
end

def download_file(file, sha256)
  require "open-uri"

  url = "https://github.com/ankane/ml-builds/releases/download/xgboost-1.4.0/#{file}"
  puts "Downloading #{file}..."
  contents = URI.open(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  dest = "vendor/#{file}"
  File.binwrite(dest, contents)
  puts "Saved #{dest}"
end

namespace :vendor do
  task :linux do
    download_file("libxgboost.so", "5da56b56fac6c396f62dabd83d9dc1a6b38e839e31d7e5679079c575ecfa991a")
    download_file("libxgboost.arm64.so", "104e23553f49ffeb12cd19fcbfd6687d64b912bdbfac46980d5a6726a794e661")
  end

  task :mac do
    download_file("libxgboost.dylib", "f7b0594bcd1042acaff3c6e3c6ec8fae01ec4b4c1cf56cf3b0c0795d06c6c5ce")
    download_file("libxgboost.arm64.dylib", "56e2212a16419725979f67e5c0f8c5a7ab5391699ebccf1e96158c9ef725862b")
  end

  task :windows do
    download_file("xgboost.dll", "5306f3617622ecfdb09ffb734aecc53b1ef234ffc1a207034aa38281c5449315")
  end

  task all: [:linux, :mac, :windows]

  task :platform do
    if Gem.win_platform?
      Rake::Task["vendor:windows"].invoke
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      Rake::Task["vendor:mac"].invoke
    else
      Rake::Task["vendor:linux"].invoke
    end
  end
end
