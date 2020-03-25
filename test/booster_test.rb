require_relative "test_helper"

class BoosterTest < Minitest::Test
  def test_dump_text
    assert_match(/0:\[f5</, booster.dump.join)
    assert_match(/0:\[feat5</, booster_with_feature_names.dump.join)
  end

  def test_dump_json
    booster_dump = booster.dump(dump_format: "json").first
    assert JSON.parse(booster_dump)
    assert_equal 5, JSON.parse(booster_dump).fetch("split")

    feature_booster_dump = booster_with_feature_names.dump(dump_format: "json").first
    assert JSON.parse(feature_booster_dump)
    assert_equal "feat5", JSON.parse(feature_booster_dump).fetch("split")
  end

  def test_dump_model_text
    booster.dump_model(tempfile)
    assert File.exist?(tempfile)
  end

  def test_dump_model_json
    booster.dump_model(tempfile, dump_format: "json")
    assert File.exist?(tempfile)
    assert JSON.parse(File.read(tempfile))
  end

  def test_score
    expected = {"rm" => 42, "lstat" => 30, "crim" => 30, "dis" => 19, "ptratio" => 16, "age" => 21, "indus" => 7, "tax" => 13, "b" => 16, "rad" => 4, "nox" => 6, "zn" => 2, "chas" => 1}
    assert_equal expected.values.sort, booster.score.values.sort
  end

  def test_fscore
    assert_equal booster.score, booster.fscore
  end

  def test_attributes
    assert_nil booster["foo"]
    assert_equal({}, booster.attributes)

    booster["foo"] = "bar"

    assert_equal "bar", booster["foo"]
    assert_equal({ "foo" => "bar" }, booster.attributes)

    booster["foo"] = "baz"

    assert_equal "baz", booster["foo"]
    assert_equal({ "foo" => "baz" }, booster.attributes)

    booster["bar"] = "qux"

    assert_equal({ "foo" => "baz", "bar" => "qux" }, booster.attributes)

    booster["foo"] = nil

    refute_includes(booster.attributes, "foo")
  end

  private

  def load_booster
    XGBoost::Booster.new(model_file: "test/support/boston.model")
  end

  def booster
    @booster ||= load_booster
  end

  def booster_with_feature_names
    @booster_with_feature_names ||= load_booster.tap do |booster|
      booster.feature_names = (0...13).map { |idx| "feat#{idx}" }
    end
  end
end
