require_relative "test_helper"

class BoosterTest < Minitest::Test
  def text_dump_text
    assert booster.dump
  end

  def test_dump_json
    assert JSON.parse(booster.dump(dump_format: "json").first)
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

  def test_attr_not_exist
    assert_raises(ArgumentError, %(Unknown attr name: "foo")) do
      booster["foo"]
    end
  end

  def test_attr_set_once
    booster["foo"] = "bar"

    assert_equal "bar", booster["foo"]
  end

  def test_attr_set_twice
    booster["foo"] = "bar"
    booster["foo"] = "baz"

    assert_equal "baz", booster["foo"]
  end

  def test_attr_set_nil
    booster["foo"] = "bar"

    assert_includes(booster.attr_names, "foo")

    booster["foo"] = nil

    refute_includes(booster.attr_names, "foo")
  end

  def test_attr_names_when_none
    assert_equal [], booster.attr_names
  end

  def test_attr_names_with_one
    booster["foo"] = "bar"

    assert_equal ["foo"], booster.attr_names
  end

  def test_attr_names_with_two
    booster["foo"] = "bar"
    booster["bar"] = "baz"

    assert_equal ["foo", "bar"].sort, booster.attr_names.sort
  end

  private

  def booster
    @booster ||= Xgb::Booster.new(model_file: "test/support/boston.model")
  end
end
