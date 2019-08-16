require_relative "test_helper"

class RegressorTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = boston_data

    model = Xgb::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [28.509018, 25.23551, 24.38023, 32.31889, 33.371517, 27.57522]
    expected.zip(y_pred) do |exp, act|
      assert_in_delta exp, act
    end
  end
end