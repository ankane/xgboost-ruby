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

  url = "https://github.com/ankane/ml-builds/releases/download/xgboost-1.5.0/#{file}"
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
    download_file("libxgboost.so", "c1e5c8f9bffeb15e40b5f1c5db7b0208d0fd0fbf01d37df68da44eaecb3e240c")
    download_file("libxgboost.arm64.so", "b08238e175f09d18ddb5a090a3974626cfc1c1ac63545f185c01277ba218bbd6")
  end

  task :mac do
    download_file("libxgboost.dylib", "9129797ef9ea9968d1ea4f24e9b8fc81ba55647c6456d61d22e08bc337d2f21a")
    download_file("libxgboost.arm64.dylib", "710b0649c58a0e189432ba638c4b855f90d7e243e4e9f9c00837cf110a8074f3")
  end

  task :windows do
    download_file("xgboost.dll", "f26127f82e9f07a0c7e826d18809d96e8ed2874d035836163ba439501f2865bf")
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
