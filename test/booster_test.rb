require_relative "test_helper"

class BoosterTest < Minitest::Test
  def test_dump_text
    assert_match(/0:\[f2</, booster.dump.join)
    assert_match(/0:\[feat2</, booster_with_feature_names.dump.join)
  end

  def test_dump_json
    booster_dump = booster.dump(dump_format: "json").first
    assert JSON.parse(booster_dump)
    assert_equal "f2", JSON.parse(booster_dump).fetch("split")

    feature_booster_dump = booster_with_feature_names.dump(dump_format: "json").first
    assert JSON.parse(feature_booster_dump)
    assert_equal "feat2", JSON.parse(feature_booster_dump).fetch("split")
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
    expected = {"f0" => 118, "f2" => 93, "f1" => 104, "f3" => 43}
    assert_equal expected.values.sort, booster.score.values.sort
  end

  def test_fscore
    assert_equal booster.score, booster.fscore
  end

  def test_attributes
    default_attributes = {}

    assert_nil booster["foo"]
    assert_equal(default_attributes, booster.attributes)

    booster["foo"] = "bar"

    assert_equal "bar", booster["foo"]
    assert_equal(default_attributes.merge("foo" => "bar"), booster.attributes)

    booster["foo"] = "baz"

    assert_equal "baz", booster["foo"]
    assert_equal(default_attributes.merge("foo" => "baz"), booster.attributes)

    booster["bar"] = "qux"

    assert_equal(default_attributes.merge("foo" => "baz", "bar" => "qux"), booster.attributes)

    booster["foo"] = nil

    refute_includes(booster.attributes, "foo")
  end

  def test_copy
    booster.dup
    booster.clone
  end

  private

  def load_booster
    XGBoost::Booster.new(model_file: "test/support/model.bin")
  end

  def booster
    @booster ||= load_booster
  end

  def booster_with_feature_names
    @booster_with_feature_names ||= load_booster.tap do |booster|
      booster.feature_names = 4.times.map { |idx| "feat#{idx}" }
    end
  end
end
