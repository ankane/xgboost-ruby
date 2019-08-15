require_relative "test_helper"

class BoosterTest < Minitest::Test
  def text_dump_text
    assert booster.dump
  end

  def test_dump_json
    assert JSON.parse(booster.dump(dump_format: "json").first)
  end

  def test_dump_model_text
    booster.dump_model("/tmp/boston.txt")
    assert File.exist?("/tmp/boston.txt")
  end

  def test_dump_model_json
    booster.dump_model("/tmp/boston.json", dump_format: "json")
    assert File.exist?("/tmp/boston.json")
    assert JSON.parse(File.read("/tmp/boston.json"))
  end

  def test_score
    expected = {"rm" => 42, "lstat" => 30, "crim" => 30, "dis" => 19, "ptratio" => 16, "age" => 21, "indus" => 7, "tax" => 13, "b" => 16, "rad" => 4, "nox" => 6, "zn" => 2, "chas" => 1}
    assert_equal expected.values.sort, booster.score.values.sort
  end

  def test_fscore
    assert_equal booster.score, booster.fscore
  end

  private

  def booster
    @booster ||= Xgb::Booster.new(model_file: "test/support/boston.model")
  end
end
