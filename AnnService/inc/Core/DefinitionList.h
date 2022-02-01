// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef DefineVectorValueType

DefineVectorValueType(Int8, std::int8_t)
DefineVectorValueType(UInt8, std::uint8_t)
DefineVectorValueType(Int16, std::int16_t)
DefineVectorValueType(Float, float)

#endif // DefineVectorValueType

#ifdef DefineVectorValueType2

DefineVectorValueType2(Int8, Int8, std::int8_t, std::int8_t)
DefineVectorValueType2(Int8, UInt8, std::int8_t, std::uint8_t)
DefineVectorValueType2(Int8, Int16, std::int8_t, std::int16_t)
DefineVectorValueType2(Int8, Float, std::int8_t, float)
DefineVectorValueType2(UInt8, Int8, std::uint8_t, std::int8_t)
DefineVectorValueType2(UInt8, UInt8, std::uint8_t, std::uint8_t)
DefineVectorValueType2(UInt8, Int16, std::uint8_t, std::int16_t)
DefineVectorValueType2(UInt8, Float, std::uint8_t, float)
DefineVectorValueType2(Int16, Int8, std::int16_t, std::int8_t)
DefineVectorValueType2(Int16, UInt8, std::int16_t, std::uint8_t)
DefineVectorValueType2(Int16, Int16, std::int16_t, std::int16_t)
DefineVectorValueType2(Int16, Float, std::int16_t, float)
DefineVectorValueType2(Float, Int8, float, std::int8_t)
DefineVectorValueType2(Float, UInt8, float, std::uint8_t)
DefineVectorValueType2(Float, Int16, float, std::int16_t)
DefineVectorValueType2(Float, Float, float, float)

#endif // DefineVectorValueType2

#ifdef DefineDistCalcMethod

DefineDistCalcMethod(L2)
DefineDistCalcMethod(Cosine)
DefineDistCalcMethod(InnerProduct)

#endif // DefineDistCalcMethod

#ifdef DefineQuantizerType

DefineQuantizerType(None, std::shared_ptr<void>)
DefineQuantizerType(PQQuantizer, std::shared_ptr<SPTAG::COMMON::PQQuantizer>)
DefineQuantizerType(OPQQuantizer, std::shared_ptr<SPTAG::COMMON::OPQQuantizer>)

#endif // DefineQuantizerType


#ifdef DefineErrorCode

// 0x0000 ~ 0x0FFF  General Status
DefineErrorCode(Success, 0x0000)
DefineErrorCode(Fail, 0x0001)
DefineErrorCode(FailedOpenFile, 0x0002)
DefineErrorCode(FailedCreateFile, 0x0003)
DefineErrorCode(ParamNotFound, 0x0010)
DefineErrorCode(FailedParseValue, 0x0011)
DefineErrorCode(MemoryOverFlow, 0x0012)
DefineErrorCode(LackOfInputs, 0x0013)
DefineErrorCode(VectorNotFound, 0x0014)
DefineErrorCode(EmptyIndex, 0x0015)
DefineErrorCode(EmptyData, 0x0016)
DefineErrorCode(DimensionSizeMismatch, 0x0017)
DefineErrorCode(ExternalAbort, 0x0018)
DefineErrorCode(EmptyDiskIO, 0x0019)
DefineErrorCode(DiskIOFail, 0x0020)

// 0x1000 ~ 0x1FFF  Index Build Status

// 0x2000 ~ 0x2FFF  Index Serve Status

// 0x3000 ~ 0x3FFF  Helper Function Status
DefineErrorCode(ReadIni_FailedParseSection, 0x3000)
DefineErrorCode(ReadIni_FailedParseParam, 0x3001)
DefineErrorCode(ReadIni_DuplicatedSection, 0x3002)
DefineErrorCode(ReadIni_DuplicatedParam, 0x3003)


// 0x4000 ~ 0x4FFF Socket Library Status
DefineErrorCode(Socket_FailedResolveEndPoint, 0x4000)
DefineErrorCode(Socket_FailedConnectToEndPoint, 0x4001)


#endif // DefineErrorCode



#ifdef DefineIndexAlgo

DefineIndexAlgo(BKT)
DefineIndexAlgo(KDT)
DefineIndexAlgo(SPANN)

#endif // DefineIndexAlgo

// target vectors and queries
#ifdef DefineVectorFileType

// number of vectors(int32_t), dimension(int32_t)
// 1st vector
// 2nd vector
// ..
DefineVectorFileType(DEFAULT)
// dimension of 1st vector(int32_t), 1st vector
// dimension of 2nd vector(int32_t), 2nd vector
// ...
DefineVectorFileType(XVEC)
// vectors that have names and are viewable
DefineVectorFileType(TXT)

#endif // DefineVectorFileType

#ifdef DefineTruthFileType

// 1st nn id(int32_t), SPACE, 2nd nn id, SPACE, 3rd nn id,... 
// 1st nn id, SPACE, 2nd nn id, SPACE, 3rd nn id,... 
// ...
DefineTruthFileType(TXT)
// K of 1st vector(int32_t), 1st nn id(int32_t), SPACE, 2nd nn id, SPACE, 3rd nn id,... 
// K of 2nd vector(int32_t), 1st nn id, SPACE, 2nd nn id, SPACE, 3rd nn id,... 
// ...
DefineTruthFileType(XVEC)
// row(int32_t), column(int32_t), data...
DefineTruthFileType(DEFAULT)

#endif // DefineTruthFileType