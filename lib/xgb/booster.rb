module Xgb
  class Booster
    attr_accessor :best_iteration

    def initialize(params: nil, model_file: nil)
      @handle = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGBoosterCreate(nil, 0, @handle)
      if model_file
        check_result FFI.XGBoosterLoadModel(handle_pointer, model_file)
      end

      self.best_iteration = 0
      set_param(params)
      @num_class = (params && params[:num_class]) || 1
    end

    def update(dtrain, iteration)
      check_result FFI.XGBoosterUpdateOneIter(handle_pointer, iteration, dtrain.handle_pointer)
    end

    def eval_set(evals, iteration)
      dmats = ::FFI::MemoryPointer.new(:pointer, evals.size)
      dmats.write_array_of_pointer(evals.map { |v| v[0].handle_pointer })

      evnames = ::FFI::MemoryPointer.new(:pointer, evals.size)
      evnames.write_array_of_pointer(evals.map { |v| ::FFI::MemoryPointer.from_string(v[1]) })

      out_result = ::FFI::MemoryPointer.new(:pointer)

      check_result FFI.XGBoosterEvalOneIter(handle_pointer, iteration, dmats, evnames, evals.size, out_result)

      out_result.read_pointer.read_string
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
      out_len = ::FFI::MemoryPointer.new(:ulong)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGBoosterPredict(handle_pointer, data.handle_pointer, 0, ntree_limit, out_len, out_result)
      out = out_result.read_pointer.read_array_of_float(out_len.read_ulong)
      out = out.each_slice(@num_class).to_a if @num_class > 1
      out
    end

    def save_model(fname)
      check_result FFI.XGBoosterSaveModel(handle_pointer, fname)
    end

    def dump(fmap: "", with_stats: false, dump_format: "text")
      out_len = ::FFI::MemoryPointer.new(:ulong)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGBoosterDumpModelEx(handle_pointer, fmap, with_stats ? 1 : 0, dump_format, out_len, out_result)
      out_result.read_pointer.get_array_of_string(0, out_len.read_ulong).first
    end

    def dump_model(fout, fmap: "", with_stats: false, dump_format: "text")
      File.write(fout, dump(fmap: fmap, with_stats: with_stats, dump_format: dump_format))
    end

    def score
    end

    private

    def handle_pointer
      @handle.read_pointer
    end

    include Utils
  end
end
