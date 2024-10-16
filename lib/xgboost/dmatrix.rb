module XGBoost
  class DMatrix
    include Utils

    attr_reader :data, :feature_names, :feature_types, :handle

    def initialize(data, label: nil, weight: nil, missing: Float::NAN)
      @data = data

      if @data.is_a?(::FFI::AutoPointer)
        @handle = @data
        return
      end

      if data
        if matrix?(data)
          nrow = data.row_count
          ncol = data.column_count
          flat_data = data.to_a.flatten
        elsif daru?(data)
          nrow, ncol = data.shape
          flat_data = data.map_rows(&:to_a).flatten
          @feature_names = data.each_vector.map(&:name)
          @feature_types =
            data.each_vector.map(&:db_type).map do |v|
              case v
              when "INTEGER"
                "int"
              when "DOUBLE"
                "float"
              else
                raise Error, "Unknown feature type: #{v}"
              end
            end
        elsif numo?(data)
          nrow, ncol = data.shape
        elsif rover?(data)
          nrow, ncol = data.shape
          @feature_names = data.keys
          data = data.to_numo
        else
          nrow = data.count
          ncol = data.first.count
          if !data.all? { |r| r.size == ncol }
            raise ArgumentError, "Rows have different sizes"
          end
          flat_data = data.flatten
        end

        c_data = ::FFI::MemoryPointer.new(:float, nrow * ncol)
        if numo?(data)
          c_data.write_bytes(data.cast_to(Numo::SFloat).to_string)
        else
          handle_missing(flat_data, missing)
          c_data.write_array_of_float(flat_data)
        end

        out = ::FFI::MemoryPointer.new(:pointer)
        check_call FFI.XGDMatrixCreateFromMat(c_data, nrow, ncol, missing, out)
        @handle = ::FFI::AutoPointer.new(out.read_pointer, FFI.method(:XGDMatrixFree))

        @feature_names ||= ncol.times.map { |i| "f#{i}" }
      end

      self.label = label if label
      self.weight = weight if weight
    end

    def save_binary(fname, silent: true)
      check_call FFI.XGDMatrixSaveBinary(handle, fname, silent ? 1 : 0)
    end

    def label=(label)
      set_float_info("label", label)
    end

    def weight=(weight)
      set_float_info("weight", weight)
    end

    def group=(group)
      c_data = ::FFI::MemoryPointer.new(:int, group.size)
      c_data.write_array_of_int(group)
      check_call FFI.XGDMatrixSetUIntInfo(handle, "group", c_data, group.size)
    end

    def label
      float_info("label")
    end

    def weight
      float_info("weight")
    end

    def num_row
      out = ::FFI::MemoryPointer.new(:uint64)
      check_call FFI.XGDMatrixNumRow(handle, out)
      read_uint64(out)
    end

    def num_col
      out = ::FFI::MemoryPointer.new(:uint64)
      check_call FFI.XGDMatrixNumCol(handle, out)
      read_uint64(out)
    end

    def slice(rindex)
      idxset = ::FFI::MemoryPointer.new(:int, rindex.count)
      idxset.write_array_of_int(rindex)
      out = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGDMatrixSliceDMatrix(handle, idxset, rindex.size, out)

      handle = ::FFI::AutoPointer.new(out.read_pointer, FFI.method(:XGDMatrixFree))
      DMatrix.new(handle)
    end

    private

    def set_float_info(field, data)
      data = data.to_a unless data.is_a?(Array)
      c_data = ::FFI::MemoryPointer.new(:float, data.size)
      c_data.write_array_of_float(data)
      check_call FFI.XGDMatrixSetFloatInfo(handle, field.to_s, c_data, data.size)
    end

    def float_info(field)
      num_row ||= num_row()
      out_len = ::FFI::MemoryPointer.new(:int)
      out_dptr = ::FFI::MemoryPointer.new(:float, num_row)
      check_call FFI.XGDMatrixGetFloatInfo(handle, field, out_len, out_dptr)
      out_dptr.read_pointer.read_array_of_float(num_row)
    end

    def matrix?(data)
      defined?(Matrix) && data.is_a?(Matrix)
    end

    def daru?(data)
      defined?(Daru::DataFrame) && data.is_a?(Daru::DataFrame)
    end

    def numo?(data)
      defined?(Numo::NArray) && data.is_a?(Numo::NArray)
    end

    def rover?(data)
      defined?(Rover::DataFrame) && data.is_a?(Rover::DataFrame)
    end

    def handle_missing(data, missing)
      data.map! { |v| v.nil? ? missing : v }
    end
  end
end
