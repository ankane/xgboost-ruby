module XGBoost
  module Utils
    private

    def check_call(err)
      if err != 0
        # make friendly
        message = FFI.XGBGetLastError.split("\n").first.split(/:\d+: /, 2).last
        raise XGBoost::Error, message
      end
    end

    def array_of_pointers(values)
      arr = ::FFI::MemoryPointer.new(:pointer, values.size)
      arr.write_array_of_pointer(values)
      # keep reference for string pointers
      arr.instance_variable_set(:@xgboost_ref, values)
      arr
    end

    def string_pointer(value)
      ::FFI::MemoryPointer.from_string(value.to_s)
    end

    def from_cstr_to_rbstr(data, length)
      data.read_pointer.read_array_of_pointer(length.read_uint64).map do |ptr|
        ptr.read_string.force_encoding(Encoding::UTF_8)
      end
    end
  end
end
