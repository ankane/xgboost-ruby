require_relative "test_helper"

class BoosterTest < Minitest::Test
  def test_dump_model_text
    skip
    booster.dump_model("/tmp/boston.txt")
    assert File.exist?("/tmp/boston.txt")
  end

  def test_dump_model_json
    skip
    booster.dump_model("/tmp/boston.json")
    assert File.exist?("/tmp/boston.json")
    assert JSON.parse(File.read("/tmp/boston.json"))
  end

  private

  def booster
    @booster ||= Xgb::Booster.new(model_file: "test/support/boston.model")
  end
end
