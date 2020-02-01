require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    # results inconsistent between Mac and Linux
    # but consistent for Python and Ruby on same platform
    # numbers below are for Mac
    skip if ci?

    x_train, y_train, x_test, _ = iris_data
    group = [20, 80]

    model = XGBoost::Ranker.new
    model.fit(x_train, y_train, group)
    y_pred = model.predict(x_test)
    expected = [5.3243084, 6.2180243, -4.9917116, 1.140212, 0.31584498, 1.2587957]
    assert_elements_in_delta expected, y_pred[0, 6]

    model.save_model(tempfile)

    model = XGBoost::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_feature_importances
    # results inconsistent between Mac and Linux
    # but consistent for Python and Ruby on same platform
    # numbers below are for Mac
    skip if ci?

    x_train, y_train, _, _ = iris_data
    group = [20, 80]

    model = XGBoost::Ranker.new
    model.fit(x_train, y_train, group)

    expected = [0.07063404, 0.0789692, 0.41161776, 0.438779]
    assert_elements_in_delta expected, model.feature_importances
  end

  def ci?
    ENV["CI"]
  end
end
