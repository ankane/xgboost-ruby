require_relative "test_helper"

class DMatrixTest < Minitest::Test
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
end
