module XGBoost
  class DMatrix
    include Utils

    attr_reader :handle

    def initialize(data, label: nil, weight: nil, missing: Float::NAN)
      if data.is_a?(::FFI::AutoPointer)
        @handle = data
        return
      end

      if matrix?(data)
        nrow = data.row_count
        ncol = data.column_count
        flat_data = data.to_a.flatten
      elsif daru?(data)
        nrow, ncol = data.shape
        flat_data = data.map_rows(&:to_a).flatten
        feature_names = data.each_vector.map(&:name)
        feature_types =
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
        feature_names = data.keys
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

      self.feature_names = feature_names || ncol.times.map { |i| "f#{i}" }
      self.feature_types = feature_types if feature_types

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

    def num_nonmissing
      out = ::FFI::MemoryPointer.new(:uint64)
      check_call FFI.XGDMatrixNumNonMissing(handle, out)
      read_uint64(out)
    end

    def data_split_mode
      out = ::FFI::MemoryPointer.new(:uint64)
      check_call FFI.XGDMatrixDataSplitMode(handle, out)
      read_uint64(out) == 0 ? :row : :col
    end

    def slice(rindex)
      idxset = ::FFI::MemoryPointer.new(:int, rindex.count)
      idxset.write_array_of_int(rindex)
      out = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGDMatrixSliceDMatrix(handle, idxset, rindex.size, out)

      handle = ::FFI::AutoPointer.new(out.read_pointer, FFI.method(:XGDMatrixFree))
      DMatrix.new(handle)
    end

    def feature_names
      length = ::FFI::MemoryPointer.new(:uint64)
      sarr = ::FFI::MemoryPointer.new(:pointer)
      check_call(
        FFI.XGDMatrixGetStrFeatureInfo(
          handle,
          "feature_name",
          length,
          sarr
        )
      )
      feature_names = from_cstr_to_rbstr(sarr, length)
      feature_names.empty? ? nil : feature_names
    end

    def feature_names=(feature_names)
      if feature_names.nil?
        check_call(
          FFI.XGDMatrixSetStrFeatureInfo(
            handle, "feature_name", nil, 0
          )
        )
        return
      end

      # validate feature name
      feature_names =
        validate_feature_info(
          feature_names,
          num_col,
          data_split_mode == :col,
          "feature names"
        )
      if feature_names.length != feature_names.uniq.length
        raise ArgumentError, "feature_names must be unique"
      end

      # prohibit the use symbols that may affect parsing. e.g. []<
      if !feature_names.all? { |f| f.is_a?(String) && !["[", "]", "<"].any? { |x| f.include?(x) } }
        raise ArgumentError, "feature_names must be string, and may not contain [, ] or <"
      end

      c_feature_names = array_of_pointers(feature_names.map { |f| string_pointer(f) })
      check_call(
        FFI.XGDMatrixSetStrFeatureInfo(
          handle,
          "feature_name",
          c_feature_names,
          feature_names.length
        )
      )
    end

    def feature_types
      length = ::FFI::MemoryPointer.new(:uint64)
      sarr = ::FFI::MemoryPointer.new(:pointer)
      check_call(
        FFI.XGDMatrixGetStrFeatureInfo(
          handle,
          "feature_type",
          length,
          sarr
        )
      )
      res = from_cstr_to_rbstr(sarr, length)
      res.empty? ? nil : res
    end

    def feature_types=(feature_types)
      if feature_types.nil?
        check_call(
          FFI.XGDMatrixSetStrFeatureInfo(
            handle, "feature_type", nil, 0
          )
        )
        return
      end

      feature_types =
        validate_feature_info(
          feature_types,
          num_col,
          data_split_mode == :col,
          "feature types"
        )

      c_feature_types = array_of_pointers(feature_types.map { |f| string_pointer(f) })
      check_call(
        FFI.XGDMatrixSetStrFeatureInfo(
          handle,
          "feature_type",
          c_feature_types,
          feature_types.length
        )
      )
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

    def validate_feature_info(feature_info, n_features, is_column_split, name)
      if !feature_info.is_a?(Array)
        raise TypeError, "Expecting an array of strings for #{name}, got: #{feature_info.class.name}"
      end
      if feature_info.length != n_features && n_features != 0 && !is_column_split
        msg = (
          "#{name} must have the same length as the number of data columns, " +
          "expected #{n_features}, got #{feature_info.length}"
        )
        raise ArgumentError, msg
      end
      feature_info
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
