module Xgb
  module FFI
    extend ::FFI::Library

    begin
      ffi_lib Xgb.ffi_lib
    rescue LoadError => e
      raise e # if ENV["XGB_DEBUG"]
      raise LoadError, "Could not find XGBoost"
    end

    # https://github.com/dmlc/xgboost/blob/master/include/xgboost/c_api.h
    # keep same order

    # error
    attach_function :XGBGetLastError, %i[], :string

    # dmatrix
    attach_function :XGDMatrixCreateFromMat, %i[pointer uint64 uint64 float pointer], :int
    attach_function :XGDMatrixSetGroup, %i[pointer pointer uint64], :int
    attach_function :XGDMatrixNumRow, %i[pointer pointer], :int
    attach_function :XGDMatrixNumCol, %i[pointer pointer], :int
    attach_function :XGDMatrixSliceDMatrix, %i[pointer pointer uint64 pointer], :int
    attach_function :XGDMatrixFree, %i[pointer], :int
    attach_function :XGDMatrixSaveBinary, %i[pointer string int], :int
    attach_function :XGDMatrixSetFloatInfo, %i[pointer string pointer uint64], :int
    attach_function :XGDMatrixGetFloatInfo, %i[pointer string pointer pointer], :int

    # booster
    attach_function :XGBoosterCreate, %i[pointer int pointer], :int
    attach_function :XGBoosterUpdateOneIter, %i[pointer int pointer], :int
    attach_function :XGBoosterEvalOneIter, %i[pointer int pointer pointer uint64 pointer], :int
    attach_function :XGBoosterFree, %i[pointer], :int
    attach_function :XGBoosterSetParam, %i[pointer string string], :int
    attach_function :XGBoosterPredict, %i[pointer pointer int int pointer pointer], :int
    attach_function :XGBoosterLoadModel, %i[pointer string], :int
    attach_function :XGBoosterSaveModel, %i[pointer string], :int
    attach_function :XGBoosterDumpModelEx, %i[pointer string int string pointer pointer], :int
    attach_function :XGBoosterGetAttr, %i[pointer pointer pointer pointer], :int
    attach_function :XGBoosterSetAttr, %i[pointer pointer pointer], :int
    attach_function :XGBoosterGetAttrNames, %i[pointer pointer pointer], :int
  end
end
