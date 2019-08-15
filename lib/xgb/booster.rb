module Xgb
  class Booster
    def initialize(params: nil, model_file: nil)
      @handle = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGBoosterCreate(nil, 0, @handle)
      if model_file
        check_result FFI.XGBoosterLoadModel(handle_pointer, model_file)
      end

      set_param(params)
      @num_class = (params && params[:num_class]) || 1
    end

    def update(dtrain, iteration)
      check_result FFI.XGBoosterUpdateOneIter(handle_pointer, iteration, dtrain.handle_pointer)
    end

    def set_param(params, value = nil)
      if params.is_a?(Enumerable)
        params.each do |k, v|
          check_result FFI.XGBoosterSetParam(handle_pointer, k.to_s, v.to_s)
        end
      else
        check_result FFI.XGBoosterSetParam(handle_pointer, params.to_s, value.to_s)
      end
    end

    def predict(data, ntree_limit: nil)
      ntree_limit ||= 0
      out_len = ::FFI::MemoryPointer.new(:long)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGBoosterPredict(handle_pointer, data.handle_pointer, 0, ntree_limit, out_len, out_result)
      out = out_result.read_pointer.read_array_of_float(out_len.read_long)
      out = out.each_slice(@num_class).to_a if @num_class > 1
      out
    end

    def save_model(fname)
      check_result FFI.XGBoosterSaveModel(handle_pointer, fname)
    end

    private

    def handle_pointer
      @handle.read_pointer
    end

    include Utils
  end
end
