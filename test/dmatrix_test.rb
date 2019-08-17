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
    boston.save_binary(tempfile)
    assert File.exist?(tempfile)
  end

  def test_matrix
    data = Matrix.build(3, 3) { |row, col| row + col }
    label = Vector.elements([4, 5, 6])
    Xgb::DMatrix.new(data, label: label)
  end

  def test_daru_data_frame
    data = Daru::DataFrame.from_csv("test/support/boston.csv")
    label = data["medv"]
    data = data.delete_vector("medv")
    Xgb::DMatrix.new(data, label: label)
  end

  def test_numo_narray
    data = Numo::DFloat.new(3, 5).seq
    label = Numo::DFloat.new(3).seq
    Xgb::DMatrix.new(data, label: label)
  end
end
