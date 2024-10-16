module XGBoost
  class Booster
    include Utils

    def initialize(params: nil, cache: nil, model_file: nil)
      cache ||= []
      cache.each do |d|
        if !d.is_a?(DMatrix)
          raise TypeError, "invalid cache item: #{d.class.name}"
        end
      end

      dmats = array_of_pointers(cache.map { |d| d.handle_pointer })
      @handle = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGBoosterCreate(dmats, cache.length, @handle)
      ObjectSpace.define_finalizer(@handle, self.class.finalize(handle_pointer.to_i))

      cache.each do |d|
        assign_dmatrix_features(d)
      end

      if model_file
        check_call FFI.XGBoosterLoadModel(handle_pointer, model_file)
      end

      set_param(params)
    end

    def self.finalize(addr)
      # must use proc instead of stabby lambda
      proc { FFI.XGBoosterFree(::FFI::Pointer.new(:pointer, addr)) }
    end

    # TODO slice for non-string keys
    def [](key_name)
      attr(key_name)
    end

    def []=(key_name, raw_value)
      set_attr(**{key_name => raw_value})
    end

    def save_config
      length = ::FFI::MemoryPointer.new(:uint64)
      json_string = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGBoosterSaveJsonConfig(handle_pointer, length, json_string)
      json_string.read_pointer.read_string(read_uint64(length)).force_encoding(Encoding::UTF_8)
    end

    def attr(key_name)
      key = string_pointer(key_name.to_s)
      success = ::FFI::MemoryPointer.new(:int)
      out_result = ::FFI::MemoryPointer.new(:pointer)

      check_call FFI.XGBoosterGetAttr(handle_pointer, key, out_result, success)

      success.read_int == 1 ? out_result.read_pointer.read_string : nil
    end

    def attributes
      out_len = ::FFI::MemoryPointer.new(:uint64)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGBoosterGetAttrNames(handle_pointer, out_len, out_result)

      len = read_uint64(out_len)
      key_names = len.zero? ? [] : out_result.read_pointer.get_array_of_string(0, len)

      key_names.to_h { |key_name| [key_name, self[key_name]] }
    end

    def set_attr(**kwargs)
      kwargs.each do |key_name, raw_value|
        key = string_pointer(key_name)
        value = raw_value.nil? ? nil : string_pointer(raw_value.to_s)

        check_call FFI.XGBoosterSetAttr(handle_pointer, key, value)
      end
    end

    def feature_types
      get_feature_info("feature_type")
    end

    def feature_types=(features)
      set_feature_info(features, "feature_type")
    end

    def feature_names
      get_feature_info("feature_name")
    end

    def feature_names=(features)
      set_feature_info(features, "feature_name")
    end

    def set_param(params, value = nil)
      if params.is_a?(Enumerable)
        params.each do |k, v|
          check_call FFI.XGBoosterSetParam(handle_pointer, k.to_s, v.to_s)
        end
      else
        check_call FFI.XGBoosterSetParam(handle_pointer, params.to_s, value.to_s)
      end
    end

    def update(dtrain, iteration)
      check_call FFI.XGBoosterUpdateOneIter(handle_pointer, iteration, dtrain.handle_pointer)
    end

    def eval_set(evals, iteration)
      dmats = array_of_pointers(evals.map { |v| v[0].handle_pointer })
      evnames = array_of_pointers(evals.map { |v| string_pointer(v[1]) })

      out_result = ::FFI::MemoryPointer.new(:pointer)

      check_call FFI.XGBoosterEvalOneIter(handle_pointer, iteration, dmats, evnames, evals.size, out_result)

      out_result.read_pointer.read_string
    end

    def predict(data, ntree_limit: nil)
      ntree_limit ||= 0
      out_len = ::FFI::MemoryPointer.new(:uint64)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGBoosterPredict(handle_pointer, data.handle_pointer, 0, ntree_limit, 0, out_len, out_result)
      out = out_result.read_pointer.read_array_of_float(read_uint64(out_len))
      num_class = out.size / data.num_row
      out = out.each_slice(num_class).to_a if num_class > 1
      out
    end

    def save_model(fname)
      check_call FFI.XGBoosterSaveModel(handle_pointer, fname)
    end

    def best_iteration
      attr(:best_iteration)&.to_i
    end

    def best_iteration=(iteration)
      set_attr(best_iteration: iteration)
    end

    def best_score
      attr(:best_score)&.to_f
    end

    def best_score=(score)
      set_attr(best_score: score)
    end

    def num_boosted_rounds
      rounds = ::FFI::MemoryPointer.new(:int)
      check_call FFI.XGBoosterBoostedRounds(handle_pointer, rounds)
      rounds.read_int
    end

    def dump_model(fout, fmap: "", with_stats: false, dump_format: "text")
      ret = dump(fmap: fmap, with_stats: with_stats, dump_format: dump_format)
      File.open(fout, "wb") do |f|
        if dump_format == "json"
          f.print("[\n")
          ret.each_with_index do |r, i|
            f.print(r)
            f.print(",\n") if i < ret.size - 1
          end
          f.print("\n]")
        else
          ret.each_with_index do |r, i|
            f.print("booster[#{i}]:\n")
            f.print(r)
          end
        end
      end
    end

    # returns an array of strings
    def dump(fmap: "", with_stats: false, dump_format: "text")
      out_len = ::FFI::MemoryPointer.new(:uint64)
      out_result = ::FFI::MemoryPointer.new(:pointer)

      names = feature_names || []
      fnames = array_of_pointers(names.map { |fname| string_pointer(fname) })
      ftypes = array_of_pointers(feature_types || Array.new(names.size, string_pointer("float")))

      check_call FFI.XGBoosterDumpModelExWithFeatures(handle_pointer, names.size, fnames, ftypes, with_stats ? 1 : 0, dump_format, out_len, out_result)

      out_result.read_pointer.get_array_of_string(0, read_uint64(out_len))
    end

    def fscore(fmap: "")
      # always weight
      score(fmap: fmap, importance_type: "weight")
    end

    def score(fmap: "", importance_type: "weight")
      if importance_type == "weight"
        trees = dump(fmap: fmap, with_stats: false)
        fmap = {}
        trees.each do |tree|
          tree.split("\n").each do |line|
            arr = line.split("[")
            next if arr.size == 1

            fid = arr[1].split("]")[0].split("<")[0]
            fmap[fid] ||= 0
            fmap[fid] += 1
          end
        end
        fmap
      else
        average_over_splits = true
        if importance_type == "total_gain"
          importance_type = "gain"
          average_over_splits = false
        elsif importance_type == "total_cover"
          importance_type = "cover"
          average_over_splits = false
        end

        trees = dump(fmap: fmap, with_stats: true)

        importance_type += "="
        fmap = {}
        gmap = {}
        trees.each do |tree|
          tree.split("\n").each do |line|
            arr = line.split("[")
            next if arr.size == 1

            fid = arr[1].split("]")

            g = fid[1].split(importance_type)[1].split(",")[0].to_f

            fid = fid[0].split("<")[0]

            fmap[fid] ||= 0
            gmap[fid] ||= 0

            fmap[fid] += 1
            gmap[fid] += g
          end
        end

        if average_over_splits
          gmap.each_key do |fid|
            gmap[fid] = gmap[fid] / fmap[fid]
          end
        end

        gmap
      end
    end

    private

    def handle_pointer
      @handle.read_pointer
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

    def assign_dmatrix_features(data)
      if data.num_row == 0
        return
      end

      fn = data.feature_names
      ft = data.feature_types

      if feature_names.nil?
        self.feature_names = fn
      end
      if feature_types.nil?
        self.feature_types = ft
      end
    end

    def get_feature_info(field)
      length = ::FFI::MemoryPointer.new(:uint64)
      sarr = ::FFI::MemoryPointer.new(:pointer)
      if @handle.nil?
        return nil
      end
      check_call(
        FFI.XGBoosterGetStrFeatureInfo(
          handle_pointer,
          field,
          length,
          sarr
        )
      )
      feature_info = from_cstr_to_rbstr(sarr, length)
      !feature_info.empty? ? feature_info : nil
    end

    def from_cstr_to_rbstr(data, length)
      res = []
      read_uint64(length).times do |i|
        res << data.read_pointer[i * ::FFI::Pointer.size].read_pointer.read_string.force_encoding(Encoding::UTF_8)
      end
      res
    end

    def set_feature_info(features, field)
      if !features.nil?
        if !features.is_a?(Array)
          raise TypeError, "features must be an array"
        end
        c_feature_info = array_of_pointers(features.map { |f| string_pointer(f) })
        check_call(
          FFI.XGBoosterSetStrFeatureInfo(
            handle_pointer,
            field,
            c_feature_info,
            features.length
          )
        )
      else
        check_call(
          FFI.XGBoosterSetStrFeatureInfo(
            handle_pointer, field, nil, 0
          )
        )
      end
    end
  end
end
