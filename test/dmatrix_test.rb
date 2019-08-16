require_relative "test_helper"

class DMatrixTest < Minitest::Test
  def test_label
    data = [[1, 2], [3, 4]]
    label = [1, 2]
    dataset = Xgb::DMatrix.new(data, label: label)
    assert label, dataset.label
  end

  def test_weight
    data = [[1, 2], [3, 4]]
    weight = [1, 2]
    dataset = Xgb::DMatrix.new(data, weight: weight)
    assert weight, dataset.weight
  end

  def test_num_row
    assert_equal 506, boston.num_row
  end

  def test_num_col
    assert_equal 13, boston.num_col
  end

  def test_save_binary
    boston.save_binary("/tmp/dtrain.bin")
    assert File.exist?("/tmp/dtrain.bin")
  end

  def test_matrix
    data = Matrix.build(3, 3) { |row, col| row + col }
    Xgb::DMatrix.new(data, label: Matrix.column_vector([4, 5, 6]))
  end
end
