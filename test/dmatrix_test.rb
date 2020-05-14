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
    assert_equal 300, regression_train.num_row
  end

  def test_num_col
    assert_equal 4, regression_train.num_col
  end

  def test_save_binary
    regression_train.save_binary(tempfile)
    assert File.exist?(tempfile)
  end

  def test_matrix
    data = Matrix.build(3, 3) { |row, col| row + col }
    label = Vector.elements([4, 5, 6])
    XGBoost::DMatrix.new(data, label: label)
  end

  def test_daru
    data = Daru::DataFrame.from_csv(data_path)
    label = data["y"]
    data = data.delete_vector("y")
    dataset = XGBoost::DMatrix.new(data, label: label)
    names = ["x0", "x1", "x2", "x3"]
    assert_equal names, dataset.feature_names
    types = ["float", "float", "float", "int"]
    assert_equal types, dataset.feature_types
  end

  def test_numo
    skip if RUBY_PLATFORM == "java"

    require "numo/narray"
    data = Numo::DFloat.new(3, 5).seq
    label = Numo::DFloat.new(3).seq
    XGBoost::DMatrix.new(data, label: label)
  end

  def test_rover
    skip if RUBY_PLATFORM == "java"

    require "rover"
    data = Rover.read_csv(data_path)
    label = data.delete("y")
    dataset = XGBoost::DMatrix.new(data, label: label)
    names = ["x0", "x1", "x2", "x3"]
    assert_equal names, dataset.feature_names
    # TODO add types
    # types = ["float", "float", "float", "int"]
    # assert_equal types, dataset.feature_types
  end
end
