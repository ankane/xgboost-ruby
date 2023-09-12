require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    skip "Inconsistent results on different platforms" if ENV["CI"]

    x_train, y_train, x_test, _ = binary_data
    group = [100, 200]

    model = XGBoost::Ranker.new
    model.fit(x_train, y_train, group)
    y_pred = model.predict(x_test)
    expected = [-1.2509069442749023, 1.5171653032302856, 1.5171653032302856, 1.5171653032302856, 1.5171653032302856, 1.5171653032302856]
    assert_elements_in_delta expected, y_pred.first(6)

    expected = [0.0, 0.19046767055988312, 0.8095322847366333, 0.0]
    assert_elements_in_delta expected, model.feature_importances

    model.save_model(tempfile)

    model = XGBoost::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end
end
