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

    # read_uint64 not available on JRuby
    def read_uint64(ptr)
      ptr.read_array_of_uint64(1).first
    end
  end
end
