require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false
end

def download_file(target, sha256)
  version = "1.6.1"

  require "fileutils"
  require "open-uri"
  require "tmpdir"

  file = "xgboost-#{version}-#{target}.zip"
  url = "https://github.com/ankane/ml-builds/releases/download/xgboost-#{version}/#{file}"
  puts "Downloading #{file}..."
  contents = URI.open(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  Dir.chdir(Dir.mktmpdir) do
    File.binwrite(file, contents)
    dest = File.expand_path("vendor/#{target}", __dir__)
    FileUtils.rm_r(dest) if Dir.exist?(dest)
    # run apt install unzip on Linux
    system "unzip", "-q", file, "-d", dest, exception: true
  end
end

namespace :vendor do
  task :linux do
    download_file("x86_64-linux", "cb2972ef63c2a80411e801b40f0d0387a3455d376f27b7e8c8eae9106b4653a4")
    download_file("aarch64-linux", "e6428ae7e3833d6a57bbce647328c468915228a4a5db8c50d053b78686ae89ed")
  end

  task :mac do
    download_file("x86_64-darwin", "23b4aa67f4f50ae2c6e4f4610bc002ef728d596d507d4c530df497add00ff762")
    download_file("aarch64-darwin", "c0ea037a8ded16cc69173ac3bf9fc329f6ce486a4b617bb70be8094824403901")
  end

  task :windows do
    download_file("x86_64-windows", "3c60b036cabb8dc37ff60af4d849d1fda3a93e45b017c8f0dc28c9b392d4dc64")
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
