module Xgb
  module FFI
    extend ::FFI::Library
    ffi_lib ["xgboost"]

    # https://github.com/dmlc/xgboost/blob/master/include/xgboost/c_api.h
    # keep same order

    # error
    attach_function :XGBGetLastError, %i[], :string

    # dmatrix
    attach_function :XGDMatrixCreateFromMat, %i[pointer long long float pointer], :int
    attach_function :XGDMatrixNumCol, %i[pointer pointer], :int
    attach_function :XGDMatrixSetFloatInfo, %i[pointer string pointer long], :int

    # booster
    attach_function :XGBoosterCreate, %i[pointer int pointer], :int
    attach_function :XGBoosterUpdateOneIter, %i[pointer int pointer], :int
    attach_function :XGBoosterSetParam, %i[pointer string string], :int
    attach_function :XGBoosterPredict, %i[pointer pointer int int pointer pointer], :int
    attach_function :XGBoosterLoadModel, %i[pointer string], :int
    attach_function :XGBoosterSaveModel, %i[pointer string], :int
  end
end
