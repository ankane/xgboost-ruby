module Xgb
  class DMatrix
    attr_reader :data, :label, :weight

    def initialize(data, label: nil, weight: nil, missing: Float::NAN)
      @data = data
      @label = label
      @weight = weight

      c_data = ::FFI::MemoryPointer.new(:float, data.count * data.first.count)
      c_data.put_array_of_float(0, data.flatten)
      @handle = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGDMatrixCreateFromMat(c_data, data.count, data.first.count, missing, @handle)

      set_float_info("label", label) if label
    end

    def num_col
      out = ::FFI::MemoryPointer.new(:long)
      FFI.XGDMatrixNumCol(handle_pointer, out)
      out.read_long
    end

    def handle_pointer
      @handle.read_pointer
    end

    private

    def set_float_info(field, data)
      c_data = ::FFI::MemoryPointer.new(:float, data.count)
      c_data.put_array_of_float(0, data)
      check_result FFI.XGDMatrixSetFloatInfo(handle_pointer, field.to_s, c_data, data.size)
    end

    include Utils
  end
end
