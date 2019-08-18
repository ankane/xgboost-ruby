module Xgb
  module FFI
    extend ::FFI::Library

    begin
      ffi_lib ["xgboost"]
    rescue LoadError => e
      raise LoadError, "Could not find XGBoost"
    end

    # https://github.com/dmlc/xgboost/blob/master/include/xgboost/c_api.h
    # keep same order

    # error
    attach_function :XGBGetLastError, %i[], :string

    # dmatrix
    attach_function :XGDMatrixCreateFromMat, %i[pointer ulong ulong float pointer], :int
    attach_function :XGDMatrixNumRow, %i[pointer pointer], :int
    attach_function :XGDMatrixNumCol, %i[pointer pointer], :int
    attach_function :XGDMatrixSliceDMatrix, %i[pointer pointer ulong pointer], :int
    attach_function :XGDMatrixFree, %i[pointer], :int
    attach_function :XGDMatrixSaveBinary, %i[pointer string int], :int
    attach_function :XGDMatrixSetFloatInfo, %i[pointer string pointer ulong], :int
    attach_function :XGDMatrixGetFloatInfo, %i[pointer string pointer pointer], :int

    # booster
    attach_function :XGBoosterCreate, %i[pointer int pointer], :int
    attach_function :XGBoosterUpdateOneIter, %i[pointer int pointer], :int
    attach_function :XGBoosterEvalOneIter, %i[pointer int pointer pointer ulong pointer], :int
    attach_function :XGBoosterFree, %i[pointer], :int
    attach_function :XGBoosterSetParam, %i[pointer string string], :int
    attach_function :XGBoosterPredict, %i[pointer pointer int int pointer pointer], :int
    attach_function :XGBoosterLoadModel, %i[pointer string], :int
    attach_function :XGBoosterSaveModel, %i[pointer string], :int
    attach_function :XGBoosterDumpModelEx, %i[pointer string int string pointer pointer], :int
  end
end
