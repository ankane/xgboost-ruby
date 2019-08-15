module Xgb
  module Utils
    private

    def check_result(err)
      if err != 0
        # make friendly
        message = FFI.XGBGetLastError.split("\n").first.split(/:\d+: /, 2).last
        raise Xgb::Error, message
      end
    end
  end
end
