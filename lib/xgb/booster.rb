module Xgb
  class Booster
    attr_accessor :best_iteration, :feature_names

    def initialize(params: nil, model_file: nil)
      @handle = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGBoosterCreate(nil, 0, @handle)
      ObjectSpace.define_finalizer(self, self.class.finalize(handle_pointer))

      if model_file
        check_result FFI.XGBoosterLoadModel(handle_pointer, model_file)
      end

      self.best_iteration = 0
      set_param(params)
    end

    def self.finalize(pointer)
      # must use proc instead of stabby lambda
      proc { FFI.XGBoosterFree(pointer) }
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
      out_len = ::FFI::MemoryPointer.new(:uint64)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGBoosterPredict(handle_pointer, data.handle_pointer, 0, ntree_limit, out_len, out_result)
      out = out_result.read_pointer.read_array_of_float(read_uint64(out_len))
      num_class = out.size / data.num_row
      out = out.each_slice(num_class).to_a if num_class > 1
      out
    end

    def save_model(fname)
      check_result FFI.XGBoosterSaveModel(handle_pointer, fname)
    end

    # returns an array of strings
    def dump(fmap: "", with_stats: false, dump_format: "text")
      out_len = ::FFI::MemoryPointer.new(:uint64)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.XGBoosterDumpModelEx(handle_pointer, fmap, with_stats ? 1 : 0, dump_format, out_len, out_result)
      out_result.read_pointer.get_array_of_string(0, read_uint64(out_len))
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

    include Utils
  end
end
