module XGBoost
  module FFI
    extend ::FFI::Library

    begin
      ffi_lib XGBoost.ffi_lib
    rescue LoadError => e
      if ["/usr/local", "/opt/homebrew"].any? { |v| e.message.include?("Library not loaded: #{v}/opt/libomp/lib/libomp.dylib") } && e.message.include?("Reason: image not found")
        raise LoadError, "OpenMP not found. Run `brew install libomp`"
      else
        raise e
      end
    end

    # https://github.com/dmlc/xgboost/blob/master/include/xgboost/c_api.h
    # keep same order

    # general
    attach_function :XGBoostVersion, %i[pointer pointer pointer], :void
    attach_function :XGBGetLastError, %i[], :string

    # dmatrix
    attach_function :XGDMatrixCreateFromMat, %i[pointer uint64 uint64 float pointer], :int
    attach_function :XGDMatrixSetUIntInfo, %i[pointer string pointer uint64], :int
    attach_function :XGDMatrixSetStrFeatureInfo, %i[pointer string pointer uint64], :int
    attach_function :XGDMatrixGetStrFeatureInfo, %i[pointer string pointer pointer], :int
    attach_function :XGDMatrixNumRow, %i[pointer pointer], :int
    attach_function :XGDMatrixNumCol, %i[pointer pointer], :int
    attach_function :XGDMatrixNumNonMissing, %i[pointer pointer], :int
    attach_function :XGDMatrixDataSplitMode, %i[pointer pointer], :int
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
    attach_function :XGBoosterReset, %i[pointer], :int
    attach_function :XGBoosterBoostedRounds, %i[pointer pointer], :int
    attach_function :XGBoosterSetParam, %i[pointer string string], :int
    attach_function :XGBoosterGetNumFeature, %i[pointer pointer], :int
    attach_function :XGBoosterPredict, %i[pointer pointer int int int pointer pointer], :int
    attach_function :XGBoosterLoadModel, %i[pointer string], :int
    attach_function :XGBoosterSaveModel, %i[pointer string], :int
    attach_function :XGBoosterSaveJsonConfig, %i[pointer pointer pointer], :int
    attach_function :XGBoosterDumpModelExWithFeatures, %i[pointer int pointer pointer int string pointer pointer], :int
    attach_function :XGBoosterGetAttr, %i[pointer string pointer pointer], :int
    attach_function :XGBoosterSetAttr, %i[pointer string string], :int
    attach_function :XGBoosterGetAttrNames, %i[pointer pointer pointer], :int
    attach_function :XGBoosterSetStrFeatureInfo, %i[pointer string pointer uint64], :int
    attach_function :XGBoosterGetStrFeatureInfo, %i[pointer string pointer pointer], :int
  end
end
