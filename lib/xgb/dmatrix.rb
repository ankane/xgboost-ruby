module Xgb
  class DMatrix
    attr_reader :data

    def initialize(data, label: nil, weight: nil, missing: Float::NAN)
      @data = data

      @handle = ::FFI::MemoryPointer.new(:pointer)

      if data
        if matrix?(data)
          nrow = data.row_count
          ncol = data.column_count
          flat_data = data.to_a.flatten
        else
          nrow = data.count
          ncol = data.first.count
          flat_data = data.flatten
        end

        c_data = ::FFI::MemoryPointer.new(:float, nrow * ncol)
        c_data.put_array_of_float(0, flat_data)
        check_result FFI.XGDMatrixCreateFromMat(c_data, nrow, ncol, missing, @handle)
      end

      set_float_info("label", label) if label
      set_float_info("weight", weight) if weight
    end

    def label
      float_info("label")
    end

    def weight
      float_info("weight")
    end

    def num_row
      out = ::FFI::MemoryPointer.new(:ulong)
      check_result FFI.XGDMatrixNumRow(handle_pointer, out)
      out.read_ulong
    end

    def num_col
      out = ::FFI::MemoryPointer.new(:ulong)
      check_result FFI.XGDMatrixNumCol(handle_pointer, out)
      out.read_ulong
    end

    def slice(rindex)
      res = DMatrix.new(nil)
      idxset = ::FFI::MemoryPointer.new(:int, rindex.count)
      idxset.put_array_of_int(0, rindex)
      check_result FFI.XGDMatrixSliceDMatrix(handle_pointer, idxset, rindex.size, res.handle)
      res
    end

    def save_binary(fname, silent: true)
      check_result FFI.XGDMatrixSaveBinary(handle_pointer, fname, silent ? 1 : 0)
    end

    def handle
      @handle
    end

    def handle_pointer
      @handle.read_pointer
    end

    private

    def set_float_info(field, data)
      data = matrix?(data) ? data.to_a[0] : data
      c_data = ::FFI::MemoryPointer.new(:float, data.count)
      c_data.put_array_of_float(0, data)
      check_result FFI.XGDMatrixSetFloatInfo(handle_pointer, field.to_s, c_data, data.size)
    end

    def float_info(field)
      num_row ||= num_row()
      out_len = ::FFI::MemoryPointer.new(:int)
      out_dptr = ::FFI::MemoryPointer.new(:float, num_row)
      check_result FFI.XGDMatrixGetFloatInfo(handle_pointer, field, out_len, out_dptr)
      out_dptr.read_pointer.read_array_of_float(num_row)
    end

    def matrix?(data)
      defined?(Matrix) && data.is_a?(Matrix)
    end

    include Utils
  end
end
