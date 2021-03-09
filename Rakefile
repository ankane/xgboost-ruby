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

  url = "https://github.com/ankane/ml-builds/releases/download/xgboost-1.3.0/#{file}"
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
    download_file("libxgboost.so", "b68d1b1b435196faa96d55687de37caa7e31d15be6457051b3809e5f55276027")
  end

  task :mac do
    download_file("libxgboost.dylib", "5dfa148e4b4c74050c18c4e90016d6c410ce51c7ecbaaa4bd0d1f3068857d646")
    download_file("libxgboost.arm64.dylib", "aa239a9d9fb25fc1d12d6bf240bf7346bf1c8f45c14dde769b49ce314e0ff81d")
  end

  task :windows do
    download_file("xgboost.dll", "20dfd4832f89a178ed02ac343b807bdab5eb2d8405e2f647d325901892060b65")
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
