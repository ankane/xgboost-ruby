require_relative "test_helper"

class DMatrixTest < Minitest::Test
  def test_label
    data = [[1, 2], [3, 4]]
    label = [1, 2]
    dataset = XGBoost::DMatrix.new(data, label: label)
    assert label, dataset.label
  end

  def test_weight
    data = [[1, 2], [3, 4]]
    weight = [1, 2]
    dataset = XGBoost::DMatrix.new(data, weight: weight)
    assert weight, dataset.weight
  end

  def test_feature_names_and_types
    data = [[1, 2], [3, 4]]
    label = [1, 2]
    dataset = XGBoost::DMatrix.new(data, label: label)
    assert_equal ["f0", "f1"], dataset.feature_names
    assert_nil dataset.feature_types
  end

  def test_num_row
    assert_equal 506, boston.num_row
  end

  def test_num_col
    assert_equal 13, boston.num_col
  end

  def test_save_binary
    boston.save_binary(tempfile)
    assert File.exist?(tempfile)
  end

  def test_matrix
    data = Matrix.build(3, 3) { |row, col| row + col }
    label = Vector.elements([4, 5, 6])
    XGBoost::DMatrix.new(data, label: label)
  end

  def test_daru_data_frame
    data = Daru::DataFrame.from_csv("test/data/boston/boston.csv")
    label = data["medv"]
    data = data.delete_vector("medv")
    dataset = XGBoost::DMatrix.new(data, label: label)
    names = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]
    types = ["float", "float", "float", "int", "float", "float", "float", "float", "int", "float", "float", "float", "float"]
    assert_equal names, dataset.feature_names
    assert_equal types, dataset.feature_types
  end

  def test_numo_narray
    skip if RUBY_PLATFORM == "java"

    require "numo/narray"
    data = Numo::DFloat.new(3, 5).seq
    label = Numo::DFloat.new(3).seq
    XGBoost::DMatrix.new(data, label: label)
  end
end
