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

      dmats = array_of_pointers(cache.map { |d| d.handle })
      out = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGBoosterCreate(dmats, cache.length, out)
      @handle = ::FFI::AutoPointer.new(out.read_pointer, FFI.method(:XGBoosterFree))

      cache.each do |d|
        assign_dmatrix_features(d)
      end

      if model_file
        check_call FFI.XGBoosterLoadModel(handle, model_file)
      end

      set_param(params)
    end

    def [](key_name)
      if key_name.is_a?(String)
        return attr(key_name)
      end

      # TODO slice

      raise TypeError, "expected string"
    end

    def []=(key_name, raw_value)
      set_attr(**{key_name => raw_value})
    end

    def save_config
      length = ::FFI::MemoryPointer.new(:uint64)
      json_string = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGBoosterSaveJsonConfig(handle, length, json_string)
      json_string.read_pointer.read_string(length.read_uint64).force_encoding(Encoding::UTF_8)
    end

    def reset
      check_call FFI.XGBoosterReset(handle)
      self
    end

    def attr(key)
      ret = ::FFI::MemoryPointer.new(:pointer)
      success = ::FFI::MemoryPointer.new(:int)
      check_call FFI.XGBoosterGetAttr(handle, key.to_s, ret, success)
      success.read_int != 0 ? ret.read_pointer.read_string : nil
    end

    def attributes
      length = ::FFI::MemoryPointer.new(:uint64)
      sarr = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGBoosterGetAttrNames(handle, length, sarr)
      attr_names = from_cstr_to_rbstr(sarr, length)
      attr_names.to_h { |n| [n, attr(n)] }
    end

    def set_attr(**kwargs)
      kwargs.each do |key, value|
        check_call FFI.XGBoosterSetAttr(handle, key.to_s, value&.to_s)
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
          check_call FFI.XGBoosterSetParam(handle, k.to_s, v.to_s)
        end
      else
        check_call FFI.XGBoosterSetParam(handle, params.to_s, value.to_s)
      end
    end

    def update(dtrain, iteration)
      check_call FFI.XGBoosterUpdateOneIter(handle, iteration, dtrain.handle)
    end

    def eval_set(evals, iteration)
      dmats = array_of_pointers(evals.map { |v| v[0].handle })
      evnames = array_of_pointers(evals.map { |v| string_pointer(v[1]) })

      out_result = ::FFI::MemoryPointer.new(:pointer)

      check_call FFI.XGBoosterEvalOneIter(handle, iteration, dmats, evnames, evals.size, out_result)

      out_result.read_pointer.read_string
    end

    def predict(data, ntree_limit: nil)
      ntree_limit ||= 0
      out_len = ::FFI::MemoryPointer.new(:uint64)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      check_call FFI.XGBoosterPredict(handle, data.handle, 0, ntree_limit, 0, out_len, out_result)
      out = out_result.read_pointer.read_array_of_float(out_len.read_uint64)
      num_class = out.size / data.num_row
      out = out.each_slice(num_class).to_a if num_class > 1
      out
    end

    def save_model(fname)
      check_call FFI.XGBoosterSaveModel(handle, fname)
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
      check_call FFI.XGBoosterBoostedRounds(handle, rounds)
      rounds.read_int
    end

    def num_features
      features = ::FFI::MemoryPointer.new(:uint64)
      check_call FFI.XGBoosterGetNumFeature(handle, features)
      features.read_uint64
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

      check_call FFI.XGBoosterDumpModelExWithFeatures(handle, names.size, fnames, ftypes, with_stats ? 1 : 0, dump_format, out_len, out_result)

      out_result.read_pointer.get_array_of_string(0, out_len.read_uint64)
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

    def handle
      @handle
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
          handle,
          field,
          length,
          sarr
        )
      )
      feature_info = from_cstr_to_rbstr(sarr, length)
      !feature_info.empty? ? feature_info : nil
    end

    def set_feature_info(features, field)
      if !features.nil?
        if !features.is_a?(Array)
          raise TypeError, "features must be an array"
        end
        c_feature_info = array_of_pointers(features.map { |f| string_pointer(f) })
        check_call(
          FFI.XGBoosterSetStrFeatureInfo(
            handle,
            field,
            c_feature_info,
            features.length
          )
        )
      else
        check_call(
          FFI.XGBoosterSetStrFeatureInfo(
            handle, field, nil, 0
          )
        )
      end
    end
  end
end
