module Xgb
  module Utils
    private

    def check_result(err)
      raise Xgb::Error, FFI.XGBGetLastError if err != 0
    end
  end
end
