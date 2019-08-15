require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "csv"
require "json"

class Minitest::Test
  private

  def boston
    @boston ||= load_csv("boston.csv")
  end

  def boston_train
    @boston_train ||= Xgb::DMatrix.new(boston.data[0...300], label: boston.label[0...300])
  end

  def boston_test
    @boston_test ||= Xgb::DMatrix.new(boston.data[300..-1], label: boston.label[300..-1])
  end

  def iris
    @iris ||= load_csv("iris.csv")
  end

  def iris_train
    @iris_train ||= Xgb::DMatrix.new(iris.data[0...100], label: iris.label[0...100])
  end

  def iris_test
    @iris_test ||= Xgb::DMatrix.new(iris.data[100..-1], label: iris.label[100..-1])
  end

  def load_csv(filename)
    x = []
    y = []
    CSV.foreach("test/support/#{filename}", headers: true).each do |row|
      row = row.to_a.map { |_, v| v.to_f }
      x << row[0..-2]
      y << row[-1]
    end
    Xgb::DMatrix.new(x, label: y)
  end
end
