module XGBoost
  class CallbackContainer
    attr_reader :callbacks, :history

    def initialize(callbacks)
      @callbacks = callbacks
      @history = {}

      callbacks.each do |callback|
        raise ArgumentError, 'callback must subclass XGBoost::TrainingCallback.' unless callback.is_a?(TrainingCallback)
      end
    end

    def before_training(model: nil)
      callbacks.each do |callback|
        model = callback.before_training(model: model)
        unless model.is_a?(XGBoost::Booster)
          raise ArgumentError, "Callback #{callback.class}#before_training must return an instance of XGBoost::Booster"
        end
      end
      model
    end

    def after_training(model: nil)
      callbacks.each do |callback|
        model = callback.after_training(model: model)
        unless model.is_a?(XGBoost::Booster)
          raise ArgumentError, "Callback #{callback.class}#after_training must return an instance of XGBoost::Booster"
        end
      end
      model
    end

    def before_iteration(model: nil, epoch: nil)
      callbacks.none? || callbacks.any? do |callback|
        callback.before_iteration(model: model, epoch: epoch)
      end
    end

    def after_iteration(model: nil, epoch: nil, res: nil)
      update_history(res)

      callbacks.none? || callbacks.any? do |callback|
        callback.after_iteration(model: model, epoch: epoch, history: history)
      end
    end

    private

    def update_history(res)
      res.each do |name, value|
        data_name, metric_name = name.split('-', 2)
        history[data_name] ||= {}
        history[data_name][metric_name] ||= []
        history[data_name][metric_name] << value
      end
    end
  end
end
