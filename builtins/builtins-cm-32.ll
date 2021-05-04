; ModuleID = 'test32.bc'
source_filename = "builtins/builtins-c-genx.cpp"
target datalayout = "e-p:32:32-i64:64-n8:16:32"
target triple = "genx32-unknown-linux-gnu"

%class.ArgWriter = type { i32, i32*, i32, i32, i64 }
%class.UniformWriter = type { i32*, i32**, i32, i64, <5 x i32> }
%class.VaryingWriter = type { i32*, i32**, i32, i64, <5 x i32> }
%"struct.PrintInfo::Encoding4Uniform" = type { i8 }
%"struct.PrintInfo::Encoding4Varying" = type { i8 }

$_ZN7details13__impl_divremILi1EEEu2CMvbT__cS1_S1_u2CMvrT__c = comdat any

$_ZN7details13__impl_divremILi2EEEu2CMvbT__cS1_S1_u2CMvrT__c = comdat any

$_ZN7details13__impl_divremILi4EEEu2CMvbT__cS1_S1_u2CMvrT__c = comdat any

$_ZN7details13__impl_divremILi8EEEu2CMvbT__cS1_S1_u2CMvrT__c = comdat any

$_ZN7details13__impl_divremILi16EEEu2CMvbT__cS1_S1_u2CMvrT__c = comdat any

$_ZN7details13__impl_divremILi32EEEu2CMvbT__cS1_S1_u2CMvrT__c = comdat any

$_ZN7details13__impl_divremILi1EEEu2CMvbT__sS1_S1_u2CMvrT__s = comdat any

$_ZN7details13__impl_divremILi2EEEu2CMvbT__sS1_S1_u2CMvrT__s = comdat any

$_ZN7details13__impl_divremILi4EEEu2CMvbT__sS1_S1_u2CMvrT__s = comdat any

$_ZN7details13__impl_divremILi8EEEu2CMvbT__sS1_S1_u2CMvrT__s = comdat any

$_ZN7details13__impl_divremILi16EEEu2CMvbT__sS1_S1_u2CMvrT__s = comdat any

$_ZN7details13__impl_divremILi32EEEu2CMvbT__sS1_S1_u2CMvrT__s = comdat any

$_ZN7details13__impl_divremILi1EEEu2CMvbT__iS1_S1_u2CMvrT__j = comdat any

$_ZN7details13__impl_divremILi2EEEu2CMvbT__iS1_S1_u2CMvrT__j = comdat any

$_ZN7details13__impl_divremILi4EEEu2CMvbT__iS1_S1_u2CMvrT__j = comdat any

$_ZN7details13__impl_divremILi8EEEu2CMvbT__iS1_S1_u2CMvrT__j = comdat any

$_ZN7details13__impl_divremILi16EEEu2CMvbT__iS1_S1_u2CMvrT__j = comdat any

$_ZN7details13__impl_divremILi32EEEu2CMvbT__iS1_S1_u2CMvrT__j = comdat any

$_ZN7details14__impl_udivremILi1EEEu2CMvbT__hS1_S1_u2CMvrT__h = comdat any

$_ZN7details14__impl_udivremILi2EEEu2CMvbT__hS1_S1_u2CMvrT__h = comdat any

$_ZN7details14__impl_udivremILi4EEEu2CMvbT__hS1_S1_u2CMvrT__h = comdat any

$_ZN7details14__impl_udivremILi8EEEu2CMvbT__hS1_S1_u2CMvrT__h = comdat any

$_ZN7details14__impl_udivremILi16EEEu2CMvbT__hS1_S1_u2CMvrT__h = comdat any

$_ZN7details14__impl_udivremILi32EEEu2CMvbT__hS1_S1_u2CMvrT__h = comdat any

$_ZN7details14__impl_udivremILi1EEEu2CMvbT__tS1_S1_u2CMvrT__t = comdat any

$_ZN7details14__impl_udivremILi2EEEu2CMvbT__tS1_S1_u2CMvrT__t = comdat any

$_ZN7details14__impl_udivremILi4EEEu2CMvbT__tS1_S1_u2CMvrT__t = comdat any

$_ZN7details14__impl_udivremILi8EEEu2CMvbT__tS1_S1_u2CMvrT__t = comdat any

$_ZN7details14__impl_udivremILi16EEEu2CMvbT__tS1_S1_u2CMvrT__t = comdat any

$_ZN7details14__impl_udivremILi32EEEu2CMvbT__tS1_S1_u2CMvrT__t = comdat any

$_ZN7details14__impl_udivremILi1EEEu2CMvbT__jS1_S1_u2CMvrT__j = comdat any

$_ZN7details14__impl_udivremILi2EEEu2CMvbT__jS1_S1_u2CMvrT__j = comdat any

$_ZN7details14__impl_udivremILi4EEEu2CMvbT__jS1_S1_u2CMvrT__j = comdat any

$_ZN7details14__impl_udivremILi8EEEu2CMvbT__jS1_S1_u2CMvrT__j = comdat any

$_ZN7details14__impl_udivremILi16EEEu2CMvbT__jS1_S1_u2CMvrT__j = comdat any

$_ZN7details14__impl_udivremILi32EEEu2CMvbT__jS1_S1_u2CMvrT__j = comdat any

$_ZN7details21_cm_print_init_offsetE15cm_surfaceindexj = comdat any

$_ZN9ArgWriterC2EjPKjiy = comdat any

$_Z14GetFormatedStrI9ArgWriterEu2CMvb128_cPKcS3_RT_ = comdat any

$_ZN7details16_cm_print_formatILi128EEEv15cm_surfaceindexju2CMvbT__c = comdat any

$_ZN7details25_cm_print_init_offset_oclEu2CMvb1_jj = comdat any

$_Z20cm_svm_scatter_writeIiLi1EEvu2CMvbT0__ju2CMvbT0__T_ = comdat any

$_Z17get_auxiliary_strv = comdat any

$_ZN9PrintInfo14switchEncodingI13UniformWriter13VaryingWriterEEbNS_8EncodingET_T0_ = comdat any

$_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i = comdat any

$_ZN13VaryingWriterC2ERjRPKjiyu2CMvb5_i = comdat any

$_Z12write_atomicIL14CmAtomicOpType0EjLi8EENSt9enable_ifIXaaaaaaaaaaneT_LS0_7EneT_LS0_18EneT_LS0_2EneT_LS0_3EneT_LS0_255EclL_ZN7detailsL10isPowerOf2EjjET1_Li32EEEvE4typeE15cm_surfaceindexu2CMvbT1__ju2CMvbT1__T0_u2CMvrT1__S6_ = comdat any

$_ZN7details18is_valid_atomic_opIjL14CmAtomicOpType0ELi8EE5checkEv = comdat any

$_ZN7details16__impl_hex2floatEj = comdat any

$_Z13CopyPlainTextILj128EEiPKcu2CMvrT__cii = comdat any

$_ZN7details7Arg2StrI9ArgWriterEEu2CMvb100_ccRT_ = comdat any

$_Z12CopyFullTextILj100ELj128EEiu2CMvrT__ciu2CMvrT0__cii = comdat any

$_ZN7details11CopyTillSepIJLc37ELc0EEPKcLj128EEEiT0_iu2CMvrT1__cii = comdat any

$_ZN7details17Arg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details17Arg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details17Arg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details17Arg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details17Arg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details17Arg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details17Arg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details17Arg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20UniArg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20VarArg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN9PrintInfo19getEncoding4UniformIbEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11uniform2StrIbEEDav = comdat any

$_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii = comdat any

$_Z12ValueAdapterIbEDaT_ = comdat any

$_ZN9ArgWriter16GetElementaryArgEv = comdat any

$_ZN7details11CopyTillSepIJLc0EEPKcLj100EEEiT0_iu2CMvrT1__cii = comdat any

$_ZN9PrintInfo19getEncoding4VaryingIbEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11varying2StrIbEEDav = comdat any

$_Z21requiredSpace4VecElemIbL9LaneState1EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIbL9LaneState1EEEiu2CMvr100_ci = comdat any

$_Z21requiredSpace4VecElemIbL9LaneState0EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIbL9LaneState0EEEiu2CMvr100_ci = comdat any

$_ZN9ArgWriter8WriteArgIbEEvv = comdat any

$_Z3maxIiET_S0_S0_ = comdat any

$_Z6strLenIPKcEiT_ = comdat any

$_ZN7details20UniArg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20VarArg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN9PrintInfo19getEncoding4UniformIiEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11uniform2StrIiEEDav = comdat any

$_Z16cmType2SpecifierIiEPKcv = comdat any

$_ZN9ArgWriter8WriteArgIiEEvv = comdat any

$_ZN9PrintInfo14type2SpecifierIiEEPKcv = comdat any

$_ZN7details18_cm_print_args_rawIiEEv15cm_surfaceindexjjj = comdat any

$_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_ = comdat any

$_ZN9PrintInfo19getEncoding4VaryingIiEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11varying2StrIiEEDav = comdat any

$_Z21requiredSpace4VecElemIiL9LaneState1EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIiL9LaneState1EEEiu2CMvr100_ci = comdat any

$_Z21requiredSpace4VecElemIiL9LaneState0EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIiL9LaneState0EEEiu2CMvr100_ci = comdat any

$_ZN7details20UniArg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20VarArg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN9PrintInfo19getEncoding4UniformIjEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11uniform2StrIjEEDav = comdat any

$_Z16cmType2SpecifierIjEPKcv = comdat any

$_ZN9ArgWriter8WriteArgIjEEvv = comdat any

$_ZN9PrintInfo14type2SpecifierIjEEPKcv = comdat any

$_ZN7details18_cm_print_args_rawIjEEv15cm_surfaceindexjjj = comdat any

$_ZN9PrintInfo19getEncoding4VaryingIjEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11varying2StrIjEEDav = comdat any

$_Z21requiredSpace4VecElemIjL9LaneState1EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIjL9LaneState1EEEiu2CMvr100_ci = comdat any

$_Z21requiredSpace4VecElemIjL9LaneState0EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIjL9LaneState0EEEiu2CMvr100_ci = comdat any

$_ZN7details20UniArg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20VarArg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN9PrintInfo19getEncoding4UniformIfEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11uniform2StrIfEEDav = comdat any

$_Z16cmType2SpecifierIfEPKcv = comdat any

$_ZN9ArgWriter8WriteArgIfEEvv = comdat any

$_ZN9PrintInfo14type2SpecifierIfEEPKcv = comdat any

$_ZN7details18_cm_print_args_rawIfEEv15cm_surfaceindexjjj = comdat any

$_ZN9PrintInfo19getEncoding4VaryingIfEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11varying2StrIfEEDav = comdat any

$_Z21requiredSpace4VecElemIfL9LaneState1EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIfL9LaneState1EEEiu2CMvr100_ci = comdat any

$_Z21requiredSpace4VecElemIfL9LaneState0EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIfL9LaneState0EEEiu2CMvr100_ci = comdat any

$_ZN7details20UniArg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20VarArg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN9PrintInfo19getEncoding4UniformIxEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11uniform2StrIxEEDav = comdat any

$_Z16cmType2SpecifierIxEPKcv = comdat any

$_ZN9ArgWriter8WriteArgIxEEvv = comdat any

$_ZN9PrintInfo14type2SpecifierIxEEPKcv = comdat any

$_ZN7details18_cm_print_args_rawIxEEv15cm_surfaceindexjjj = comdat any

$_ZN9PrintInfo19getEncoding4VaryingIxEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11varying2StrIxEEDav = comdat any

$_Z21requiredSpace4VecElemIxL9LaneState1EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIxL9LaneState1EEEiu2CMvr100_ci = comdat any

$_Z21requiredSpace4VecElemIxL9LaneState0EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIxL9LaneState0EEEiu2CMvr100_ci = comdat any

$_ZN7details20UniArg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20VarArg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN9PrintInfo19getEncoding4UniformIyEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11uniform2StrIyEEDav = comdat any

$_Z16cmType2SpecifierIyEPKcv = comdat any

$_ZN9ArgWriter8WriteArgIyEEvv = comdat any

$_ZN9PrintInfo14type2SpecifierIyEEPKcv = comdat any

$_ZN7details18_cm_print_args_rawIyEEv15cm_surfaceindexjjj = comdat any

$_ZN9PrintInfo19getEncoding4VaryingIyEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11varying2StrIyEEDav = comdat any

$_Z21requiredSpace4VecElemIyL9LaneState1EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIyL9LaneState1EEEiu2CMvr100_ci = comdat any

$_Z21requiredSpace4VecElemIyL9LaneState0EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIyL9LaneState0EEEiu2CMvr100_ci = comdat any

$_ZN7details20UniArg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20VarArg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN9PrintInfo19getEncoding4UniformIdEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11uniform2StrIdEEDav = comdat any

$_Z16cmType2SpecifierIdEPKcv = comdat any

$_ZN9ArgWriter8WriteArgIdEEvv = comdat any

$_ZN9PrintInfo14type2SpecifierIdEEPKcv = comdat any

$_ZN7details18_cm_print_args_rawIdEEv15cm_surfaceindexjjj = comdat any

$_ZN9PrintInfo19getEncoding4VaryingIdEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11varying2StrIdEEDav = comdat any

$_Z21requiredSpace4VecElemIdL9LaneState1EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIdL9LaneState1EEEiu2CMvr100_ci = comdat any

$_Z21requiredSpace4VecElemIdL9LaneState0EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIdL9LaneState0EEEiu2CMvr100_ci = comdat any

$_ZN7details20UniArg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN7details20VarArg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c = comdat any

$_ZN9PrintInfo19getEncoding4UniformIPvEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11uniform2StrIPvEEDav = comdat any

$_Z16cmType2SpecifierIPvEPKcv = comdat any

$_ZN9ArgWriter8WriteArgIPvEEvv = comdat any

$_ZN9PrintInfo19getEncoding4VaryingIPvEENS_8EncodingEv = comdat any

$_ZN9ArgWriter11varying2StrIPvEEDav = comdat any

$_Z21requiredSpace4VecElemIPvL9LaneState1EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIPvL9LaneState1EEEiu2CMvr100_ci = comdat any

$_Z21requiredSpace4VecElemIPvL9LaneState0EEiv = comdat any

$_ZN9ArgWriter19writeFormat4VecElemIPvL9LaneState0EEEiu2CMvr100_ci = comdat any

$_ZN7details11CopyTillSepIJLc0EELj100ELj128EEEiu2CMvrT0__ciu2CMvrT1__cii = comdat any

$_Z5writeIcLi128EEv15cm_surfaceindexiu2CMvbT0__T_ = comdat any

$_ZN7details25cm_svm_scatter_write_implIiLi1EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_ = comdat any

$_ZN9PrintInfo22switchEncoding4UniformI13UniformWriterEEbNS_8EncodingET_ = comdat any

$_ZN9PrintInfo22switchEncoding4VaryingI13VaryingWriterEEbNS_8EncodingET_ = comdat any

$_ZN9PrintInfo6detail14switchEncodingI13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET_T0_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIb13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIi13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIj13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIf13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIx13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIy13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingId13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIPv13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_ = comdat any

$_ZNK9PrintInfo16Encoding4Uniform4callIbEENS_8EncodingEv = comdat any

$_ZN13UniformWriter4callIbEEvv = comdat any

$_ZN13UniformWriter8WriteArgIbEEvv = comdat any

$_ZN13UniformWriter16GetElementaryArgEv = comdat any

$_ZN7details18_cm_print_type_oclIRA1_KcEENS_18SHADER_PRINTF_TYPEEv = comdat any

$_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_ = comdat any

$_ZN7details25cm_svm_scatter_write_implIjLi1EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_ = comdat any

$_ZNK9PrintInfo16Encoding4Uniform4callIiEENS_8EncodingEv = comdat any

$_ZN13UniformWriter4callIiEEvv = comdat any

$_ZN13UniformWriter8WriteArgIiEEvv = comdat any

$_ZN7details18_cm_print_type_oclIiEENS_18SHADER_PRINTF_TYPEEv = comdat any

$_ZNK9PrintInfo16Encoding4Uniform4callIjEENS_8EncodingEv = comdat any

$_ZN13UniformWriter4callIjEEvv = comdat any

$_ZN13UniformWriter8WriteArgIjEEvv = comdat any

$_ZN7details18_cm_print_type_oclIjEENS_18SHADER_PRINTF_TYPEEv = comdat any

$_ZNK9PrintInfo16Encoding4Uniform4callIfEENS_8EncodingEv = comdat any

$_ZN13UniformWriter4callIfEEvv = comdat any

$_ZN13UniformWriter8WriteArgIfEEvv = comdat any

$_ZN7details18_cm_print_type_oclIfEENS_18SHADER_PRINTF_TYPEEv = comdat any

$_ZNK9PrintInfo16Encoding4Uniform4callIxEENS_8EncodingEv = comdat any

$_ZN13UniformWriter4callIxEEvv = comdat any

$_ZN13UniformWriter8WriteArgIxEEvv = comdat any

$_ZN7details18_cm_print_type_oclIxEENS_18SHADER_PRINTF_TYPEEv = comdat any

$_Z20cm_svm_scatter_writeIjLi2EEvu2CMvbT0__ju2CMvbT0__T_ = comdat any

$_ZN7details25cm_svm_scatter_write_implIjLi2EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_ = comdat any

$_ZNK9PrintInfo16Encoding4Uniform4callIyEENS_8EncodingEv = comdat any

$_ZN13UniformWriter4callIyEEvv = comdat any

$_ZN13UniformWriter8WriteArgIyEEvv = comdat any

$_ZN7details18_cm_print_type_oclIyEENS_18SHADER_PRINTF_TYPEEv = comdat any

$_ZNK9PrintInfo16Encoding4Uniform4callIdEENS_8EncodingEv = comdat any

$_ZN13UniformWriter4callIdEEvv = comdat any

$_ZN13UniformWriter8WriteArgIdEEvv = comdat any

$_ZN7details18_cm_print_type_oclIdEENS_18SHADER_PRINTF_TYPEEv = comdat any

$_ZNK9PrintInfo16Encoding4Uniform4callIPvEENS_8EncodingEv = comdat any

$_ZN13UniformWriter4callIPvEEvv = comdat any

$_ZN13UniformWriter8WriteArgIPvEEvv = comdat any

$_ZN7details18_cm_print_type_oclIPvEENS_18SHADER_PRINTF_TYPEEv = comdat any

$_ZN9PrintInfo6detail14switchEncodingI13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET_T0_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIb13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIi13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIj13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIf13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIx13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIy13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingId13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_ = comdat any

$_ZN9PrintInfo6detail21applyIfProperEncodingIPv13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_ = comdat any

$_ZNK9PrintInfo16Encoding4Varying4callIbEENS_8EncodingEv = comdat any

$_ZN13VaryingWriter4callIbEEvv = comdat any

$_ZN13VaryingWriter8WriteArgIbEEvv = comdat any

$_ZN13VaryingWriter12WriteVecElemIbEEvv = comdat any

$_ZNK9PrintInfo16Encoding4Varying4callIiEENS_8EncodingEv = comdat any

$_ZN13VaryingWriter4callIiEEvv = comdat any

$_ZN13VaryingWriter8WriteArgIiEEvv = comdat any

$_ZN13VaryingWriter12WriteVecElemIiEEvv = comdat any

$_ZNK9PrintInfo16Encoding4Varying4callIjEENS_8EncodingEv = comdat any

$_ZN13VaryingWriter4callIjEEvv = comdat any

$_ZN13VaryingWriter8WriteArgIjEEvv = comdat any

$_ZN13VaryingWriter12WriteVecElemIjEEvv = comdat any

$_ZNK9PrintInfo16Encoding4Varying4callIfEENS_8EncodingEv = comdat any

$_ZN13VaryingWriter4callIfEEvv = comdat any

$_ZN13VaryingWriter8WriteArgIfEEvv = comdat any

$_ZN13VaryingWriter12WriteVecElemIfEEvv = comdat any

$_ZNK9PrintInfo16Encoding4Varying4callIxEENS_8EncodingEv = comdat any

$_ZN13VaryingWriter4callIxEEvv = comdat any

$_ZN13VaryingWriter8WriteArgIxEEvv = comdat any

$_ZN13VaryingWriter12WriteVecElemIxEEvv = comdat any

$_ZNK9PrintInfo16Encoding4Varying4callIyEENS_8EncodingEv = comdat any

$_ZN13VaryingWriter4callIyEEvv = comdat any

$_ZN13VaryingWriter8WriteArgIyEEvv = comdat any

$_ZN13VaryingWriter12WriteVecElemIyEEvv = comdat any

$_ZNK9PrintInfo16Encoding4Varying4callIdEENS_8EncodingEv = comdat any

$_ZN13VaryingWriter4callIdEEvv = comdat any

$_ZN13VaryingWriter8WriteArgIdEEvv = comdat any

$_ZN13VaryingWriter12WriteVecElemIdEEvv = comdat any

$_ZNK9PrintInfo16Encoding4Varying4callIPvEENS_8EncodingEv = comdat any

$_ZN13VaryingWriter4callIPvEEvv = comdat any

$_ZN13VaryingWriter8WriteArgIPvEEvv = comdat any

$_ZN13VaryingWriter12WriteVecElemIPvEEvv = comdat any

@.str = private unnamed_addr constant [3 x i8] c"((\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"))\00", align 1
@.str.2 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.3 = private unnamed_addr constant [6 x i8] c"false\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"true\00", align 1
@_ZL14OffLaneBoolStr = internal constant [10 x i8] c"_________\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.6 = private unnamed_addr constant [3 x i8] c"%u\00", align 1
@.str.7 = private unnamed_addr constant [3 x i8] c"%f\00", align 1
@.str.8 = private unnamed_addr constant [5 x i8] c"%lld\00", align 1
@.str.9 = private unnamed_addr constant [5 x i8] c"%llu\00", align 1
@.str.10 = private unnamed_addr constant [5 x i8] c"%08X\00", align 1

; Function Attrs: noinline nounwind
define internal signext i8 @_Z24__cm_intrinsic_impl_sdivcc(i8 signext, i8 signext) #0 {
  %3 = alloca i8, align 1
  %4 = alloca i8, align 1
  %5 = alloca <1 x i8>, align 1
  %6 = alloca <1 x i8>, align 1
  %7 = alloca <1 x i8>, align 1
  %8 = alloca <1 x i8>, align 1
  store i8 %0, i8* %3, align 1, !tbaa !7
  store i8 %1, i8* %4, align 1, !tbaa !7
  %9 = load i8, i8* %3, align 1, !tbaa !7
  %10 = insertelement <1 x i8> undef, i8 %9, i32 0
  %11 = shufflevector <1 x i8> %10, <1 x i8> undef, <1 x i32> zeroinitializer
  store <1 x i8> %11, <1 x i8>* %5, align 1, !tbaa !7
  %12 = load i8, i8* %4, align 1, !tbaa !7
  %13 = insertelement <1 x i8> undef, i8 %12, i32 0
  %14 = shufflevector <1 x i8> %13, <1 x i8> undef, <1 x i32> zeroinitializer
  store <1 x i8> %14, <1 x i8>* %6, align 1, !tbaa !7
  %15 = load <1 x i8>, <1 x i8>* %5, align 1, !tbaa !7
  %16 = load <1 x i8>, <1 x i8>* %6, align 1, !tbaa !7
  %17 = call <1 x i8> @_ZN7details13__impl_divremILi1EEEu2CMvbT__cS1_S1_u2CMvrT__c(<1 x i8> %15, <1 x i8> %16, <1 x i8>* %7)
  store <1 x i8> %17, <1 x i8>* %8, align 1, !tbaa !7
  %18 = load <1 x i8>, <1 x i8>* %8, align 1, !tbaa !7
  %19 = call <1 x i8> @llvm.genx.rdregioni.v1i8.v1i8.i16(<1 x i8> %18, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %20 = extractelement <1 x i8> %19, i32 0
  ret i8 %20
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i8> @_ZN7details13__impl_divremILi1EEEu2CMvbT__cS1_S1_u2CMvrT__c(<1 x i8>, <1 x i8>, <1 x i8>*) #1 comdat {
  %4 = alloca <1 x i8>, align 1
  %5 = alloca <1 x i8>, align 1
  %6 = alloca <1 x i8>*, align 1
  %7 = alloca <1 x i16>, align 2
  %8 = alloca <1 x i16>, align 2
  %9 = alloca <1 x i16>, align 2
  %10 = alloca <1 x i16>, align 2
  store <1 x i8> %0, <1 x i8>* %4, align 1, !tbaa !7
  store <1 x i8> %1, <1 x i8>* %5, align 1, !tbaa !7
  store <1 x i8>* %2, <1 x i8>** %6, align 1, !tbaa !7
  %11 = load <1 x i8>, <1 x i8>* %4, align 1, !tbaa !7
  %12 = sext <1 x i8> %11 to <1 x i16>
  store <1 x i16> %12, <1 x i16>* %7, align 2, !tbaa !7
  %13 = load <1 x i8>, <1 x i8>* %5, align 1, !tbaa !7
  %14 = sext <1 x i8> %13 to <1 x i16>
  store <1 x i16> %14, <1 x i16>* %8, align 2, !tbaa !7
  %15 = load <1 x i16>, <1 x i16>* %7, align 2, !tbaa !7
  %16 = load <1 x i16>, <1 x i16>* %8, align 2, !tbaa !7
  %17 = call <1 x i16> @_ZN7details13__impl_divremILi1EEEu2CMvbT__sS1_S1_u2CMvrT__s(<1 x i16> %15, <1 x i16> %16, <1 x i16>* %9)
  store <1 x i16> %17, <1 x i16>* %10, align 2, !tbaa !7
  %18 = load <1 x i16>, <1 x i16>* %9, align 2, !tbaa !7
  %19 = trunc <1 x i16> %18 to <1 x i8>
  store <1 x i8> %19, <1 x i8>* %2
  %20 = load <1 x i16>, <1 x i16>* %10, align 2, !tbaa !7
  %21 = trunc <1 x i16> %20 to <1 x i8>
  ret <1 x i8> %21
}

; Function Attrs: nounwind readnone
declare <1 x i8> @llvm.genx.rdregioni.v1i8.v1i8.i16(<1 x i8>, i32, i32, i32, i16, i32) #2

; Function Attrs: noinline nounwind
define internal <1 x i8> @_Z24__cm_intrinsic_impl_sdivu2CMvb1_cS_(<1 x i8>, <1 x i8>) #0 {
  %3 = alloca <1 x i8>, align 1
  %4 = alloca <1 x i8>, align 1
  %5 = alloca <1 x i8>, align 1
  %6 = alloca <1 x i8>, align 1
  store <1 x i8> %0, <1 x i8>* %3, align 1, !tbaa !7
  store <1 x i8> %1, <1 x i8>* %4, align 1, !tbaa !7
  %7 = load <1 x i8>, <1 x i8>* %3, align 1, !tbaa !7
  %8 = load <1 x i8>, <1 x i8>* %4, align 1, !tbaa !7
  %9 = call <1 x i8> @_ZN7details13__impl_divremILi1EEEu2CMvbT__cS1_S1_u2CMvrT__c(<1 x i8> %7, <1 x i8> %8, <1 x i8>* %5)
  store <1 x i8> %9, <1 x i8>* %6, align 1, !tbaa !7
  %10 = load <1 x i8>, <1 x i8>* %6, align 1, !tbaa !7
  ret <1 x i8> %10
}

; Function Attrs: noinline nounwind
define internal <2 x i8> @_Z24__cm_intrinsic_impl_sdivu2CMvb2_cS_(<2 x i8>, <2 x i8>) #3 {
  %3 = alloca <2 x i8>, align 2
  %4 = alloca <2 x i8>, align 2
  %5 = alloca <2 x i8>, align 2
  %6 = alloca <2 x i8>, align 2
  store <2 x i8> %0, <2 x i8>* %3, align 2, !tbaa !7
  store <2 x i8> %1, <2 x i8>* %4, align 2, !tbaa !7
  %7 = load <2 x i8>, <2 x i8>* %3, align 2, !tbaa !7
  %8 = load <2 x i8>, <2 x i8>* %4, align 2, !tbaa !7
  %9 = call <2 x i8> @_ZN7details13__impl_divremILi2EEEu2CMvbT__cS1_S1_u2CMvrT__c(<2 x i8> %7, <2 x i8> %8, <2 x i8>* %5)
  store <2 x i8> %9, <2 x i8>* %6, align 2, !tbaa !7
  %10 = load <2 x i8>, <2 x i8>* %6, align 2, !tbaa !7
  ret <2 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <2 x i8> @_ZN7details13__impl_divremILi2EEEu2CMvbT__cS1_S1_u2CMvrT__c(<2 x i8>, <2 x i8>, <2 x i8>*) #4 comdat {
  %4 = alloca <2 x i8>, align 2
  %5 = alloca <2 x i8>, align 2
  %6 = alloca <2 x i8>*, align 2
  %7 = alloca <2 x i16>, align 4
  %8 = alloca <2 x i16>, align 4
  %9 = alloca <2 x i16>, align 4
  %10 = alloca <2 x i16>, align 4
  store <2 x i8> %0, <2 x i8>* %4, align 2, !tbaa !7
  store <2 x i8> %1, <2 x i8>* %5, align 2, !tbaa !7
  store <2 x i8>* %2, <2 x i8>** %6, align 2, !tbaa !7
  %11 = load <2 x i8>, <2 x i8>* %4, align 2, !tbaa !7
  %12 = sext <2 x i8> %11 to <2 x i16>
  store <2 x i16> %12, <2 x i16>* %7, align 4, !tbaa !7
  %13 = load <2 x i8>, <2 x i8>* %5, align 2, !tbaa !7
  %14 = sext <2 x i8> %13 to <2 x i16>
  store <2 x i16> %14, <2 x i16>* %8, align 4, !tbaa !7
  %15 = load <2 x i16>, <2 x i16>* %7, align 4, !tbaa !7
  %16 = load <2 x i16>, <2 x i16>* %8, align 4, !tbaa !7
  %17 = call <2 x i16> @_ZN7details13__impl_divremILi2EEEu2CMvbT__sS1_S1_u2CMvrT__s(<2 x i16> %15, <2 x i16> %16, <2 x i16>* %9)
  store <2 x i16> %17, <2 x i16>* %10, align 4, !tbaa !7
  %18 = load <2 x i16>, <2 x i16>* %9, align 4, !tbaa !7
  %19 = trunc <2 x i16> %18 to <2 x i8>
  call void @llvm.genx.vstore.v2i8.p0v2i8(<2 x i8> %19, <2 x i8>* %2)
  %20 = load <2 x i16>, <2 x i16>* %10, align 4, !tbaa !7
  %21 = trunc <2 x i16> %20 to <2 x i8>
  ret <2 x i8> %21
}

; Function Attrs: noinline nounwind
define internal <4 x i8> @_Z24__cm_intrinsic_impl_sdivu2CMvb4_cS_(<4 x i8>, <4 x i8>) #5 {
  %3 = alloca <4 x i8>, align 4
  %4 = alloca <4 x i8>, align 4
  %5 = alloca <4 x i8>, align 4
  %6 = alloca <4 x i8>, align 4
  store <4 x i8> %0, <4 x i8>* %3, align 4, !tbaa !7
  store <4 x i8> %1, <4 x i8>* %4, align 4, !tbaa !7
  %7 = load <4 x i8>, <4 x i8>* %3, align 4, !tbaa !7
  %8 = load <4 x i8>, <4 x i8>* %4, align 4, !tbaa !7
  %9 = call <4 x i8> @_ZN7details13__impl_divremILi4EEEu2CMvbT__cS1_S1_u2CMvrT__c(<4 x i8> %7, <4 x i8> %8, <4 x i8>* %5)
  store <4 x i8> %9, <4 x i8>* %6, align 4, !tbaa !7
  %10 = load <4 x i8>, <4 x i8>* %6, align 4, !tbaa !7
  ret <4 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <4 x i8> @_ZN7details13__impl_divremILi4EEEu2CMvbT__cS1_S1_u2CMvrT__c(<4 x i8>, <4 x i8>, <4 x i8>*) #6 comdat {
  %4 = alloca <4 x i8>, align 4
  %5 = alloca <4 x i8>, align 4
  %6 = alloca <4 x i8>*, align 4
  %7 = alloca <4 x i16>, align 8
  %8 = alloca <4 x i16>, align 8
  %9 = alloca <4 x i16>, align 8
  %10 = alloca <4 x i16>, align 8
  store <4 x i8> %0, <4 x i8>* %4, align 4, !tbaa !7
  store <4 x i8> %1, <4 x i8>* %5, align 4, !tbaa !7
  store <4 x i8>* %2, <4 x i8>** %6, align 4, !tbaa !7
  %11 = load <4 x i8>, <4 x i8>* %4, align 4, !tbaa !7
  %12 = sext <4 x i8> %11 to <4 x i16>
  store <4 x i16> %12, <4 x i16>* %7, align 8, !tbaa !7
  %13 = load <4 x i8>, <4 x i8>* %5, align 4, !tbaa !7
  %14 = sext <4 x i8> %13 to <4 x i16>
  store <4 x i16> %14, <4 x i16>* %8, align 8, !tbaa !7
  %15 = load <4 x i16>, <4 x i16>* %7, align 8, !tbaa !7
  %16 = load <4 x i16>, <4 x i16>* %8, align 8, !tbaa !7
  %17 = call <4 x i16> @_ZN7details13__impl_divremILi4EEEu2CMvbT__sS1_S1_u2CMvrT__s(<4 x i16> %15, <4 x i16> %16, <4 x i16>* %9)
  store <4 x i16> %17, <4 x i16>* %10, align 8, !tbaa !7
  %18 = load <4 x i16>, <4 x i16>* %9, align 8, !tbaa !7
  %19 = trunc <4 x i16> %18 to <4 x i8>
  call void @llvm.genx.vstore.v4i8.p0v4i8(<4 x i8> %19, <4 x i8>* %2)
  %20 = load <4 x i16>, <4 x i16>* %10, align 8, !tbaa !7
  %21 = trunc <4 x i16> %20 to <4 x i8>
  ret <4 x i8> %21
}

; Function Attrs: noinline nounwind
define dso_local <8 x i8> @_Z24__cm_intrinsic_impl_sdivu2CMvb8_cS_(<8 x i8>, <8 x i8>) #7 {
  %3 = alloca <8 x i8>, align 8
  %4 = alloca <8 x i8>, align 8
  %5 = alloca <8 x i8>, align 8
  %6 = alloca <8 x i8>, align 8
  store <8 x i8> %0, <8 x i8>* %3, align 8, !tbaa !7
  store <8 x i8> %1, <8 x i8>* %4, align 8, !tbaa !7
  %7 = load <8 x i8>, <8 x i8>* %3, align 8, !tbaa !7
  %8 = load <8 x i8>, <8 x i8>* %4, align 8, !tbaa !7
  %9 = call <8 x i8> @_ZN7details13__impl_divremILi8EEEu2CMvbT__cS1_S1_u2CMvrT__c(<8 x i8> %7, <8 x i8> %8, <8 x i8>* %5)
  store <8 x i8> %9, <8 x i8>* %6, align 8, !tbaa !7
  %10 = load <8 x i8>, <8 x i8>* %6, align 8, !tbaa !7
  ret <8 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <8 x i8> @_ZN7details13__impl_divremILi8EEEu2CMvbT__cS1_S1_u2CMvrT__c(<8 x i8>, <8 x i8>, <8 x i8>*) #8 comdat {
  %4 = alloca <8 x i8>, align 8
  %5 = alloca <8 x i8>, align 8
  %6 = alloca <8 x i8>*, align 8
  %7 = alloca <8 x i16>, align 16
  %8 = alloca <8 x i16>, align 16
  %9 = alloca <8 x i16>, align 16
  %10 = alloca <8 x i16>, align 16
  store <8 x i8> %0, <8 x i8>* %4, align 8, !tbaa !7
  store <8 x i8> %1, <8 x i8>* %5, align 8, !tbaa !7
  store <8 x i8>* %2, <8 x i8>** %6, align 8, !tbaa !7
  %11 = load <8 x i8>, <8 x i8>* %4, align 8, !tbaa !7
  %12 = sext <8 x i8> %11 to <8 x i16>
  store <8 x i16> %12, <8 x i16>* %7, align 16, !tbaa !7
  %13 = load <8 x i8>, <8 x i8>* %5, align 8, !tbaa !7
  %14 = sext <8 x i8> %13 to <8 x i16>
  store <8 x i16> %14, <8 x i16>* %8, align 16, !tbaa !7
  %15 = load <8 x i16>, <8 x i16>* %7, align 16, !tbaa !7
  %16 = load <8 x i16>, <8 x i16>* %8, align 16, !tbaa !7
  %17 = call <8 x i16> @_ZN7details13__impl_divremILi8EEEu2CMvbT__sS1_S1_u2CMvrT__s(<8 x i16> %15, <8 x i16> %16, <8 x i16>* %9)
  store <8 x i16> %17, <8 x i16>* %10, align 16, !tbaa !7
  %18 = load <8 x i16>, <8 x i16>* %9, align 16, !tbaa !7
  %19 = trunc <8 x i16> %18 to <8 x i8>
  call void @llvm.genx.vstore.v8i8.p0v8i8(<8 x i8> %19, <8 x i8>* %2)
  %20 = load <8 x i16>, <8 x i16>* %10, align 16, !tbaa !7
  %21 = trunc <8 x i16> %20 to <8 x i8>
  ret <8 x i8> %21
}

; Function Attrs: noinline nounwind
define dso_local <16 x i8> @_Z24__cm_intrinsic_impl_sdivu2CMvb16_cS_(<16 x i8>, <16 x i8>) #9 {
  %3 = alloca <16 x i8>, align 16
  %4 = alloca <16 x i8>, align 16
  %5 = alloca <16 x i8>, align 16
  %6 = alloca <16 x i8>, align 16
  store <16 x i8> %0, <16 x i8>* %3, align 16, !tbaa !7
  store <16 x i8> %1, <16 x i8>* %4, align 16, !tbaa !7
  %7 = load <16 x i8>, <16 x i8>* %3, align 16, !tbaa !7
  %8 = load <16 x i8>, <16 x i8>* %4, align 16, !tbaa !7
  %9 = call <16 x i8> @_ZN7details13__impl_divremILi16EEEu2CMvbT__cS1_S1_u2CMvrT__c(<16 x i8> %7, <16 x i8> %8, <16 x i8>* %5)
  store <16 x i8> %9, <16 x i8>* %6, align 16, !tbaa !7
  %10 = load <16 x i8>, <16 x i8>* %6, align 16, !tbaa !7
  ret <16 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <16 x i8> @_ZN7details13__impl_divremILi16EEEu2CMvbT__cS1_S1_u2CMvrT__c(<16 x i8>, <16 x i8>, <16 x i8>*) #10 comdat {
  %4 = alloca <16 x i8>, align 16
  %5 = alloca <16 x i8>, align 16
  %6 = alloca <16 x i8>*, align 16
  %7 = alloca <16 x i16>, align 32
  %8 = alloca <16 x i16>, align 32
  %9 = alloca <16 x i16>, align 32
  %10 = alloca <16 x i16>, align 32
  store <16 x i8> %0, <16 x i8>* %4, align 16, !tbaa !7
  store <16 x i8> %1, <16 x i8>* %5, align 16, !tbaa !7
  store <16 x i8>* %2, <16 x i8>** %6, align 16, !tbaa !7
  %11 = load <16 x i8>, <16 x i8>* %4, align 16, !tbaa !7
  %12 = sext <16 x i8> %11 to <16 x i16>
  store <16 x i16> %12, <16 x i16>* %7, align 32, !tbaa !7
  %13 = load <16 x i8>, <16 x i8>* %5, align 16, !tbaa !7
  %14 = sext <16 x i8> %13 to <16 x i16>
  store <16 x i16> %14, <16 x i16>* %8, align 32, !tbaa !7
  %15 = load <16 x i16>, <16 x i16>* %7, align 32, !tbaa !7
  %16 = load <16 x i16>, <16 x i16>* %8, align 32, !tbaa !7
  %17 = call <16 x i16> @_ZN7details13__impl_divremILi16EEEu2CMvbT__sS1_S1_u2CMvrT__s(<16 x i16> %15, <16 x i16> %16, <16 x i16>* %9)
  store <16 x i16> %17, <16 x i16>* %10, align 32, !tbaa !7
  %18 = load <16 x i16>, <16 x i16>* %9, align 32, !tbaa !7
  %19 = trunc <16 x i16> %18 to <16 x i8>
  call void @llvm.genx.vstore.v16i8.p0v16i8(<16 x i8> %19, <16 x i8>* %2)
  %20 = load <16 x i16>, <16 x i16>* %10, align 32, !tbaa !7
  %21 = trunc <16 x i16> %20 to <16 x i8>
  ret <16 x i8> %21
}

; Function Attrs: noinline nounwind
define internal <32 x i8> @_Z24__cm_intrinsic_impl_sdivu2CMvb32_cS_(<32 x i8>, <32 x i8>) #11 {
  %3 = alloca <32 x i8>, align 32
  %4 = alloca <32 x i8>, align 32
  %5 = alloca <32 x i8>, align 32
  %6 = alloca <32 x i8>, align 32
  store <32 x i8> %0, <32 x i8>* %3, align 32, !tbaa !7
  store <32 x i8> %1, <32 x i8>* %4, align 32, !tbaa !7
  %7 = load <32 x i8>, <32 x i8>* %3, align 32, !tbaa !7
  %8 = load <32 x i8>, <32 x i8>* %4, align 32, !tbaa !7
  %9 = call <32 x i8> @_ZN7details13__impl_divremILi32EEEu2CMvbT__cS1_S1_u2CMvrT__c(<32 x i8> %7, <32 x i8> %8, <32 x i8>* %5)
  store <32 x i8> %9, <32 x i8>* %6, align 32, !tbaa !7
  %10 = load <32 x i8>, <32 x i8>* %6, align 32, !tbaa !7
  ret <32 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <32 x i8> @_ZN7details13__impl_divremILi32EEEu2CMvbT__cS1_S1_u2CMvrT__c(<32 x i8>, <32 x i8>, <32 x i8>*) #12 comdat {
  %4 = alloca <32 x i8>, align 32
  %5 = alloca <32 x i8>, align 32
  %6 = alloca <32 x i8>*, align 32
  %7 = alloca <32 x i16>, align 64
  %8 = alloca <32 x i16>, align 64
  %9 = alloca <32 x i16>, align 64
  %10 = alloca <32 x i16>, align 64
  store <32 x i8> %0, <32 x i8>* %4, align 32, !tbaa !7
  store <32 x i8> %1, <32 x i8>* %5, align 32, !tbaa !7
  store <32 x i8>* %2, <32 x i8>** %6, align 32, !tbaa !7
  %11 = load <32 x i8>, <32 x i8>* %4, align 32, !tbaa !7
  %12 = sext <32 x i8> %11 to <32 x i16>
  store <32 x i16> %12, <32 x i16>* %7, align 64, !tbaa !7
  %13 = load <32 x i8>, <32 x i8>* %5, align 32, !tbaa !7
  %14 = sext <32 x i8> %13 to <32 x i16>
  store <32 x i16> %14, <32 x i16>* %8, align 64, !tbaa !7
  %15 = load <32 x i16>, <32 x i16>* %7, align 64, !tbaa !7
  %16 = load <32 x i16>, <32 x i16>* %8, align 64, !tbaa !7
  %17 = call <32 x i16> @_ZN7details13__impl_divremILi32EEEu2CMvbT__sS1_S1_u2CMvrT__s(<32 x i16> %15, <32 x i16> %16, <32 x i16>* %9)
  store <32 x i16> %17, <32 x i16>* %10, align 64, !tbaa !7
  %18 = load <32 x i16>, <32 x i16>* %9, align 64, !tbaa !7
  %19 = trunc <32 x i16> %18 to <32 x i8>
  call void @llvm.genx.vstore.v32i8.p0v32i8(<32 x i8> %19, <32 x i8>* %2)
  %20 = load <32 x i16>, <32 x i16>* %10, align 64, !tbaa !7
  %21 = trunc <32 x i16> %20 to <32 x i8>
  ret <32 x i8> %21
}

; Function Attrs: noinline nounwind
define internal signext i8 @_Z24__cm_intrinsic_impl_sremcc(i8 signext, i8 signext) #0 {
  %3 = alloca i8, align 1
  %4 = alloca i8, align 1
  %5 = alloca <1 x i8>, align 1
  %6 = alloca <1 x i8>, align 1
  %7 = alloca <1 x i8>, align 1
  store i8 %0, i8* %3, align 1, !tbaa !7
  store i8 %1, i8* %4, align 1, !tbaa !7
  %8 = load i8, i8* %3, align 1, !tbaa !7
  %9 = insertelement <1 x i8> undef, i8 %8, i32 0
  %10 = shufflevector <1 x i8> %9, <1 x i8> undef, <1 x i32> zeroinitializer
  store <1 x i8> %10, <1 x i8>* %5, align 1, !tbaa !7
  %11 = load i8, i8* %4, align 1, !tbaa !7
  %12 = insertelement <1 x i8> undef, i8 %11, i32 0
  %13 = shufflevector <1 x i8> %12, <1 x i8> undef, <1 x i32> zeroinitializer
  store <1 x i8> %13, <1 x i8>* %6, align 1, !tbaa !7
  %14 = load <1 x i8>, <1 x i8>* %5, align 1, !tbaa !7
  %15 = load <1 x i8>, <1 x i8>* %6, align 1, !tbaa !7
  %16 = call <1 x i8> @_ZN7details13__impl_divremILi1EEEu2CMvbT__cS1_S1_u2CMvrT__c(<1 x i8> %14, <1 x i8> %15, <1 x i8>* %7)
  %17 = load <1 x i8>, <1 x i8>* %7, align 1, !tbaa !7
  %18 = call <1 x i8> @llvm.genx.rdregioni.v1i8.v1i8.i16(<1 x i8> %17, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %19 = extractelement <1 x i8> %18, i32 0
  ret i8 %19
}

; Function Attrs: noinline nounwind
define internal <1 x i8> @_Z24__cm_intrinsic_impl_sremu2CMvb1_cS_(<1 x i8>, <1 x i8>) #0 {
  %3 = alloca <1 x i8>, align 1
  %4 = alloca <1 x i8>, align 1
  %5 = alloca <1 x i8>, align 1
  store <1 x i8> %0, <1 x i8>* %3, align 1, !tbaa !7
  store <1 x i8> %1, <1 x i8>* %4, align 1, !tbaa !7
  %6 = load <1 x i8>, <1 x i8>* %3, align 1, !tbaa !7
  %7 = load <1 x i8>, <1 x i8>* %4, align 1, !tbaa !7
  %8 = call <1 x i8> @_ZN7details13__impl_divremILi1EEEu2CMvbT__cS1_S1_u2CMvrT__c(<1 x i8> %6, <1 x i8> %7, <1 x i8>* %5)
  %9 = load <1 x i8>, <1 x i8>* %5, align 1, !tbaa !7
  ret <1 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <2 x i8> @_Z24__cm_intrinsic_impl_sremu2CMvb2_cS_(<2 x i8>, <2 x i8>) #3 {
  %3 = alloca <2 x i8>, align 2
  %4 = alloca <2 x i8>, align 2
  %5 = alloca <2 x i8>, align 2
  store <2 x i8> %0, <2 x i8>* %3, align 2, !tbaa !7
  store <2 x i8> %1, <2 x i8>* %4, align 2, !tbaa !7
  %6 = load <2 x i8>, <2 x i8>* %3, align 2, !tbaa !7
  %7 = load <2 x i8>, <2 x i8>* %4, align 2, !tbaa !7
  %8 = call <2 x i8> @_ZN7details13__impl_divremILi2EEEu2CMvbT__cS1_S1_u2CMvrT__c(<2 x i8> %6, <2 x i8> %7, <2 x i8>* %5)
  %9 = load <2 x i8>, <2 x i8>* %5, align 2, !tbaa !7
  ret <2 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <4 x i8> @_Z24__cm_intrinsic_impl_sremu2CMvb4_cS_(<4 x i8>, <4 x i8>) #5 {
  %3 = alloca <4 x i8>, align 4
  %4 = alloca <4 x i8>, align 4
  %5 = alloca <4 x i8>, align 4
  store <4 x i8> %0, <4 x i8>* %3, align 4, !tbaa !7
  store <4 x i8> %1, <4 x i8>* %4, align 4, !tbaa !7
  %6 = load <4 x i8>, <4 x i8>* %3, align 4, !tbaa !7
  %7 = load <4 x i8>, <4 x i8>* %4, align 4, !tbaa !7
  %8 = call <4 x i8> @_ZN7details13__impl_divremILi4EEEu2CMvbT__cS1_S1_u2CMvrT__c(<4 x i8> %6, <4 x i8> %7, <4 x i8>* %5)
  %9 = load <4 x i8>, <4 x i8>* %5, align 4, !tbaa !7
  ret <4 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <8 x i8> @_Z24__cm_intrinsic_impl_sremu2CMvb8_cS_(<8 x i8>, <8 x i8>) #7 {
  %3 = alloca <8 x i8>, align 8
  %4 = alloca <8 x i8>, align 8
  %5 = alloca <8 x i8>, align 8
  store <8 x i8> %0, <8 x i8>* %3, align 8, !tbaa !7
  store <8 x i8> %1, <8 x i8>* %4, align 8, !tbaa !7
  %6 = load <8 x i8>, <8 x i8>* %3, align 8, !tbaa !7
  %7 = load <8 x i8>, <8 x i8>* %4, align 8, !tbaa !7
  %8 = call <8 x i8> @_ZN7details13__impl_divremILi8EEEu2CMvbT__cS1_S1_u2CMvrT__c(<8 x i8> %6, <8 x i8> %7, <8 x i8>* %5)
  %9 = load <8 x i8>, <8 x i8>* %5, align 8, !tbaa !7
  ret <8 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <16 x i8> @_Z24__cm_intrinsic_impl_sremu2CMvb16_cS_(<16 x i8>, <16 x i8>) #9 {
  %3 = alloca <16 x i8>, align 16
  %4 = alloca <16 x i8>, align 16
  %5 = alloca <16 x i8>, align 16
  store <16 x i8> %0, <16 x i8>* %3, align 16, !tbaa !7
  store <16 x i8> %1, <16 x i8>* %4, align 16, !tbaa !7
  %6 = load <16 x i8>, <16 x i8>* %3, align 16, !tbaa !7
  %7 = load <16 x i8>, <16 x i8>* %4, align 16, !tbaa !7
  %8 = call <16 x i8> @_ZN7details13__impl_divremILi16EEEu2CMvbT__cS1_S1_u2CMvrT__c(<16 x i8> %6, <16 x i8> %7, <16 x i8>* %5)
  %9 = load <16 x i8>, <16 x i8>* %5, align 16, !tbaa !7
  ret <16 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <32 x i8> @_Z24__cm_intrinsic_impl_sremu2CMvb32_cS_(<32 x i8>, <32 x i8>) #11 {
  %3 = alloca <32 x i8>, align 32
  %4 = alloca <32 x i8>, align 32
  %5 = alloca <32 x i8>, align 32
  store <32 x i8> %0, <32 x i8>* %3, align 32, !tbaa !7
  store <32 x i8> %1, <32 x i8>* %4, align 32, !tbaa !7
  %6 = load <32 x i8>, <32 x i8>* %3, align 32, !tbaa !7
  %7 = load <32 x i8>, <32 x i8>* %4, align 32, !tbaa !7
  %8 = call <32 x i8> @_ZN7details13__impl_divremILi32EEEu2CMvbT__cS1_S1_u2CMvrT__c(<32 x i8> %6, <32 x i8> %7, <32 x i8>* %5)
  %9 = load <32 x i8>, <32 x i8>* %5, align 32, !tbaa !7
  ret <32 x i8> %9
}

; Function Attrs: noinline nounwind
define internal signext i16 @_Z24__cm_intrinsic_impl_sdivss(i16 signext, i16 signext) #3 {
  %3 = alloca i16, align 2
  %4 = alloca i16, align 2
  %5 = alloca <1 x i16>, align 2
  %6 = alloca <1 x i16>, align 2
  %7 = alloca <1 x i16>, align 2
  %8 = alloca <1 x i16>, align 2
  store i16 %0, i16* %3, align 2, !tbaa !11
  store i16 %1, i16* %4, align 2, !tbaa !11
  %9 = load i16, i16* %3, align 2, !tbaa !11
  %10 = insertelement <1 x i16> undef, i16 %9, i32 0
  %11 = shufflevector <1 x i16> %10, <1 x i16> undef, <1 x i32> zeroinitializer
  store <1 x i16> %11, <1 x i16>* %5, align 2, !tbaa !7
  %12 = load i16, i16* %4, align 2, !tbaa !11
  %13 = insertelement <1 x i16> undef, i16 %12, i32 0
  %14 = shufflevector <1 x i16> %13, <1 x i16> undef, <1 x i32> zeroinitializer
  store <1 x i16> %14, <1 x i16>* %6, align 2, !tbaa !7
  %15 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  %16 = load <1 x i16>, <1 x i16>* %6, align 2, !tbaa !7
  %17 = call <1 x i16> @_ZN7details13__impl_divremILi1EEEu2CMvbT__sS1_S1_u2CMvrT__s(<1 x i16> %15, <1 x i16> %16, <1 x i16>* %7)
  store <1 x i16> %17, <1 x i16>* %8, align 2, !tbaa !7
  %18 = load <1 x i16>, <1 x i16>* %8, align 2, !tbaa !7
  %19 = call <1 x i16> @llvm.genx.rdregioni.v1i16.v1i16.i16(<1 x i16> %18, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %20 = extractelement <1 x i16> %19, i32 0
  ret i16 %20
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i16> @_ZN7details13__impl_divremILi1EEEu2CMvbT__sS1_S1_u2CMvrT__s(<1 x i16>, <1 x i16>, <1 x i16>*) #1 comdat {
  %4 = alloca <1 x i16>, align 2
  %5 = alloca <1 x i16>, align 2
  %6 = alloca <1 x i16>*, align 2
  %7 = alloca <1 x float>, align 4
  %8 = alloca <1 x float>, align 4
  %9 = alloca <1 x float>, align 4
  %10 = alloca <1 x float>, align 4
  %11 = alloca <1 x float>, align 4
  %12 = alloca <1 x i16>, align 2
  store <1 x i16> %0, <1 x i16>* %4, align 2, !tbaa !7
  store <1 x i16> %1, <1 x i16>* %5, align 2, !tbaa !7
  store <1 x i16>* %2, <1 x i16>** %6, align 2, !tbaa !7
  %13 = load <1 x i16>, <1 x i16>* %4, align 2, !tbaa !7
  %14 = sitofp <1 x i16> %13 to <1 x float>
  store <1 x float> %14, <1 x float>* %7, align 4, !tbaa !7
  %15 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  %16 = sitofp <1 x i16> %15 to <1 x float>
  store <1 x float> %16, <1 x float>* %8, align 4, !tbaa !7
  %17 = load <1 x float>, <1 x float>* %8, align 4, !tbaa !7
  %18 = fdiv <1 x float> <float 1.000000e+00>, %17
  store <1 x float> %18, <1 x float>* %9, align 4, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <1 x float> undef, float %19, i32 0
  %21 = shufflevector <1 x float> %20, <1 x float> undef, <1 x i32> zeroinitializer
  store <1 x float> %21, <1 x float>* %10, align 4, !tbaa !7
  %22 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %23 = load <1 x float>, <1 x float>* %10, align 4, !tbaa !7
  %24 = fmul <1 x float> %22, %23
  store <1 x float> %24, <1 x float>* %7, align 4, !tbaa !7
  %25 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %26 = load <1 x float>, <1 x float>* %9, align 4, !tbaa !7
  %27 = fmul <1 x float> %25, %26
  store <1 x float> %27, <1 x float>* %11, align 4, !tbaa !7
  %28 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %29 = fptosi <1 x float> %28 to <1 x i32>
  %30 = trunc <1 x i32> %29 to <1 x i16>
  store <1 x i16> %30, <1 x i16>* %12, align 2, !tbaa !7
  %31 = load <1 x i16>, <1 x i16>* %4, align 2, !tbaa !7
  %32 = sext <1 x i16> %31 to <1 x i32>
  %33 = load <1 x i16>, <1 x i16>* %12, align 2, !tbaa !7
  %34 = sext <1 x i16> %33 to <1 x i32>
  %35 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  %36 = sext <1 x i16> %35 to <1 x i32>
  %37 = mul <1 x i32> %34, %36
  %38 = sub <1 x i32> %32, %37
  %39 = trunc <1 x i32> %38 to <1 x i16>
  store <1 x i16> %39, <1 x i16>* %2
  %40 = load <1 x i16>, <1 x i16>* %12, align 2, !tbaa !7
  ret <1 x i16> %40
}

; Function Attrs: nounwind readnone
declare <1 x i16> @llvm.genx.rdregioni.v1i16.v1i16.i16(<1 x i16>, i32, i32, i32, i16, i32) #2

; Function Attrs: noinline nounwind
define internal <1 x i16> @_Z24__cm_intrinsic_impl_sdivu2CMvb1_sS_(<1 x i16>, <1 x i16>) #3 {
  %3 = alloca <1 x i16>, align 2
  %4 = alloca <1 x i16>, align 2
  %5 = alloca <1 x i16>, align 2
  %6 = alloca <1 x i16>, align 2
  store <1 x i16> %0, <1 x i16>* %3, align 2, !tbaa !7
  store <1 x i16> %1, <1 x i16>* %4, align 2, !tbaa !7
  %7 = load <1 x i16>, <1 x i16>* %3, align 2, !tbaa !7
  %8 = load <1 x i16>, <1 x i16>* %4, align 2, !tbaa !7
  %9 = call <1 x i16> @_ZN7details13__impl_divremILi1EEEu2CMvbT__sS1_S1_u2CMvrT__s(<1 x i16> %7, <1 x i16> %8, <1 x i16>* %5)
  store <1 x i16> %9, <1 x i16>* %6, align 2, !tbaa !7
  %10 = load <1 x i16>, <1 x i16>* %6, align 2, !tbaa !7
  ret <1 x i16> %10
}

; Function Attrs: noinline nounwind
define internal <2 x i16> @_Z24__cm_intrinsic_impl_sdivu2CMvb2_sS_(<2 x i16>, <2 x i16>) #5 {
  %3 = alloca <2 x i16>, align 4
  %4 = alloca <2 x i16>, align 4
  %5 = alloca <2 x i16>, align 4
  %6 = alloca <2 x i16>, align 4
  store <2 x i16> %0, <2 x i16>* %3, align 4, !tbaa !7
  store <2 x i16> %1, <2 x i16>* %4, align 4, !tbaa !7
  %7 = load <2 x i16>, <2 x i16>* %3, align 4, !tbaa !7
  %8 = load <2 x i16>, <2 x i16>* %4, align 4, !tbaa !7
  %9 = call <2 x i16> @_ZN7details13__impl_divremILi2EEEu2CMvbT__sS1_S1_u2CMvrT__s(<2 x i16> %7, <2 x i16> %8, <2 x i16>* %5)
  store <2 x i16> %9, <2 x i16>* %6, align 4, !tbaa !7
  %10 = load <2 x i16>, <2 x i16>* %6, align 4, !tbaa !7
  ret <2 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <2 x i16> @_ZN7details13__impl_divremILi2EEEu2CMvbT__sS1_S1_u2CMvrT__s(<2 x i16>, <2 x i16>, <2 x i16>*) #4 comdat {
  %4 = alloca <2 x i16>, align 4
  %5 = alloca <2 x i16>, align 4
  %6 = alloca <2 x i16>*, align 4
  %7 = alloca <2 x float>, align 8
  %8 = alloca <2 x float>, align 8
  %9 = alloca <2 x float>, align 8
  %10 = alloca <2 x float>, align 8
  %11 = alloca <2 x float>, align 8
  %12 = alloca <2 x i16>, align 4
  store <2 x i16> %0, <2 x i16>* %4, align 4, !tbaa !7
  store <2 x i16> %1, <2 x i16>* %5, align 4, !tbaa !7
  store <2 x i16>* %2, <2 x i16>** %6, align 4, !tbaa !7
  %13 = load <2 x i16>, <2 x i16>* %4, align 4, !tbaa !7
  %14 = sitofp <2 x i16> %13 to <2 x float>
  store <2 x float> %14, <2 x float>* %7, align 8, !tbaa !7
  %15 = load <2 x i16>, <2 x i16>* %5, align 4, !tbaa !7
  %16 = sitofp <2 x i16> %15 to <2 x float>
  store <2 x float> %16, <2 x float>* %8, align 8, !tbaa !7
  %17 = load <2 x float>, <2 x float>* %8, align 8, !tbaa !7
  %18 = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %17
  store <2 x float> %18, <2 x float>* %9, align 8, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <2 x float> undef, float %19, i32 0
  %21 = shufflevector <2 x float> %20, <2 x float> undef, <2 x i32> zeroinitializer
  store <2 x float> %21, <2 x float>* %10, align 8, !tbaa !7
  %22 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %23 = load <2 x float>, <2 x float>* %10, align 8, !tbaa !7
  %24 = fmul <2 x float> %22, %23
  store <2 x float> %24, <2 x float>* %7, align 8, !tbaa !7
  %25 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %26 = load <2 x float>, <2 x float>* %9, align 8, !tbaa !7
  %27 = fmul <2 x float> %25, %26
  store <2 x float> %27, <2 x float>* %11, align 8, !tbaa !7
  %28 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %29 = fptosi <2 x float> %28 to <2 x i32>
  %30 = trunc <2 x i32> %29 to <2 x i16>
  store <2 x i16> %30, <2 x i16>* %12, align 4, !tbaa !7
  %31 = load <2 x i16>, <2 x i16>* %4, align 4, !tbaa !7
  %32 = sext <2 x i16> %31 to <2 x i32>
  %33 = load <2 x i16>, <2 x i16>* %12, align 4, !tbaa !7
  %34 = sext <2 x i16> %33 to <2 x i32>
  %35 = load <2 x i16>, <2 x i16>* %5, align 4, !tbaa !7
  %36 = sext <2 x i16> %35 to <2 x i32>
  %37 = mul <2 x i32> %34, %36
  %38 = sub <2 x i32> %32, %37
  %39 = trunc <2 x i32> %38 to <2 x i16>
  call void @llvm.genx.vstore.v2i16.p0v2i16(<2 x i16> %39, <2 x i16>* %2)
  %40 = load <2 x i16>, <2 x i16>* %12, align 4, !tbaa !7
  ret <2 x i16> %40
}

; Function Attrs: noinline nounwind
define internal <4 x i16> @_Z24__cm_intrinsic_impl_sdivu2CMvb4_sS_(<4 x i16>, <4 x i16>) #7 {
  %3 = alloca <4 x i16>, align 8
  %4 = alloca <4 x i16>, align 8
  %5 = alloca <4 x i16>, align 8
  %6 = alloca <4 x i16>, align 8
  store <4 x i16> %0, <4 x i16>* %3, align 8, !tbaa !7
  store <4 x i16> %1, <4 x i16>* %4, align 8, !tbaa !7
  %7 = load <4 x i16>, <4 x i16>* %3, align 8, !tbaa !7
  %8 = load <4 x i16>, <4 x i16>* %4, align 8, !tbaa !7
  %9 = call <4 x i16> @_ZN7details13__impl_divremILi4EEEu2CMvbT__sS1_S1_u2CMvrT__s(<4 x i16> %7, <4 x i16> %8, <4 x i16>* %5)
  store <4 x i16> %9, <4 x i16>* %6, align 8, !tbaa !7
  %10 = load <4 x i16>, <4 x i16>* %6, align 8, !tbaa !7
  ret <4 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <4 x i16> @_ZN7details13__impl_divremILi4EEEu2CMvbT__sS1_S1_u2CMvrT__s(<4 x i16>, <4 x i16>, <4 x i16>*) #6 comdat {
  %4 = alloca <4 x i16>, align 8
  %5 = alloca <4 x i16>, align 8
  %6 = alloca <4 x i16>*, align 8
  %7 = alloca <4 x float>, align 16
  %8 = alloca <4 x float>, align 16
  %9 = alloca <4 x float>, align 16
  %10 = alloca <4 x float>, align 16
  %11 = alloca <4 x float>, align 16
  %12 = alloca <4 x i16>, align 8
  store <4 x i16> %0, <4 x i16>* %4, align 8, !tbaa !7
  store <4 x i16> %1, <4 x i16>* %5, align 8, !tbaa !7
  store <4 x i16>* %2, <4 x i16>** %6, align 8, !tbaa !7
  %13 = load <4 x i16>, <4 x i16>* %4, align 8, !tbaa !7
  %14 = sitofp <4 x i16> %13 to <4 x float>
  store <4 x float> %14, <4 x float>* %7, align 16, !tbaa !7
  %15 = load <4 x i16>, <4 x i16>* %5, align 8, !tbaa !7
  %16 = sitofp <4 x i16> %15 to <4 x float>
  store <4 x float> %16, <4 x float>* %8, align 16, !tbaa !7
  %17 = load <4 x float>, <4 x float>* %8, align 16, !tbaa !7
  %18 = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %17
  store <4 x float> %18, <4 x float>* %9, align 16, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <4 x float> undef, float %19, i32 0
  %21 = shufflevector <4 x float> %20, <4 x float> undef, <4 x i32> zeroinitializer
  store <4 x float> %21, <4 x float>* %10, align 16, !tbaa !7
  %22 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %23 = load <4 x float>, <4 x float>* %10, align 16, !tbaa !7
  %24 = fmul <4 x float> %22, %23
  store <4 x float> %24, <4 x float>* %7, align 16, !tbaa !7
  %25 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %26 = load <4 x float>, <4 x float>* %9, align 16, !tbaa !7
  %27 = fmul <4 x float> %25, %26
  store <4 x float> %27, <4 x float>* %11, align 16, !tbaa !7
  %28 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %29 = fptosi <4 x float> %28 to <4 x i32>
  %30 = trunc <4 x i32> %29 to <4 x i16>
  store <4 x i16> %30, <4 x i16>* %12, align 8, !tbaa !7
  %31 = load <4 x i16>, <4 x i16>* %4, align 8, !tbaa !7
  %32 = sext <4 x i16> %31 to <4 x i32>
  %33 = load <4 x i16>, <4 x i16>* %12, align 8, !tbaa !7
  %34 = sext <4 x i16> %33 to <4 x i32>
  %35 = load <4 x i16>, <4 x i16>* %5, align 8, !tbaa !7
  %36 = sext <4 x i16> %35 to <4 x i32>
  %37 = mul <4 x i32> %34, %36
  %38 = sub <4 x i32> %32, %37
  %39 = trunc <4 x i32> %38 to <4 x i16>
  call void @llvm.genx.vstore.v4i16.p0v4i16(<4 x i16> %39, <4 x i16>* %2)
  %40 = load <4 x i16>, <4 x i16>* %12, align 8, !tbaa !7
  ret <4 x i16> %40
}

; Function Attrs: noinline nounwind
define dso_local <8 x i16> @_Z24__cm_intrinsic_impl_sdivu2CMvb8_sS_(<8 x i16>, <8 x i16>) #9 {
  %3 = alloca <8 x i16>, align 16
  %4 = alloca <8 x i16>, align 16
  %5 = alloca <8 x i16>, align 16
  %6 = alloca <8 x i16>, align 16
  store <8 x i16> %0, <8 x i16>* %3, align 16, !tbaa !7
  store <8 x i16> %1, <8 x i16>* %4, align 16, !tbaa !7
  %7 = load <8 x i16>, <8 x i16>* %3, align 16, !tbaa !7
  %8 = load <8 x i16>, <8 x i16>* %4, align 16, !tbaa !7
  %9 = call <8 x i16> @_ZN7details13__impl_divremILi8EEEu2CMvbT__sS1_S1_u2CMvrT__s(<8 x i16> %7, <8 x i16> %8, <8 x i16>* %5)
  store <8 x i16> %9, <8 x i16>* %6, align 16, !tbaa !7
  %10 = load <8 x i16>, <8 x i16>* %6, align 16, !tbaa !7
  ret <8 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <8 x i16> @_ZN7details13__impl_divremILi8EEEu2CMvbT__sS1_S1_u2CMvrT__s(<8 x i16>, <8 x i16>, <8 x i16>*) #8 comdat {
  %4 = alloca <8 x i16>, align 16
  %5 = alloca <8 x i16>, align 16
  %6 = alloca <8 x i16>*, align 16
  %7 = alloca <8 x float>, align 32
  %8 = alloca <8 x float>, align 32
  %9 = alloca <8 x float>, align 32
  %10 = alloca <8 x float>, align 32
  %11 = alloca <8 x float>, align 32
  %12 = alloca <8 x i16>, align 16
  store <8 x i16> %0, <8 x i16>* %4, align 16, !tbaa !7
  store <8 x i16> %1, <8 x i16>* %5, align 16, !tbaa !7
  store <8 x i16>* %2, <8 x i16>** %6, align 16, !tbaa !7
  %13 = load <8 x i16>, <8 x i16>* %4, align 16, !tbaa !7
  %14 = sitofp <8 x i16> %13 to <8 x float>
  store <8 x float> %14, <8 x float>* %7, align 32, !tbaa !7
  %15 = load <8 x i16>, <8 x i16>* %5, align 16, !tbaa !7
  %16 = sitofp <8 x i16> %15 to <8 x float>
  store <8 x float> %16, <8 x float>* %8, align 32, !tbaa !7
  %17 = load <8 x float>, <8 x float>* %8, align 32, !tbaa !7
  %18 = fdiv <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %17
  store <8 x float> %18, <8 x float>* %9, align 32, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <8 x float> undef, float %19, i32 0
  %21 = shufflevector <8 x float> %20, <8 x float> undef, <8 x i32> zeroinitializer
  store <8 x float> %21, <8 x float>* %10, align 32, !tbaa !7
  %22 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %23 = load <8 x float>, <8 x float>* %10, align 32, !tbaa !7
  %24 = fmul <8 x float> %22, %23
  store <8 x float> %24, <8 x float>* %7, align 32, !tbaa !7
  %25 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %26 = load <8 x float>, <8 x float>* %9, align 32, !tbaa !7
  %27 = fmul <8 x float> %25, %26
  store <8 x float> %27, <8 x float>* %11, align 32, !tbaa !7
  %28 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %29 = fptosi <8 x float> %28 to <8 x i32>
  %30 = trunc <8 x i32> %29 to <8 x i16>
  store <8 x i16> %30, <8 x i16>* %12, align 16, !tbaa !7
  %31 = load <8 x i16>, <8 x i16>* %4, align 16, !tbaa !7
  %32 = sext <8 x i16> %31 to <8 x i32>
  %33 = load <8 x i16>, <8 x i16>* %12, align 16, !tbaa !7
  %34 = sext <8 x i16> %33 to <8 x i32>
  %35 = load <8 x i16>, <8 x i16>* %5, align 16, !tbaa !7
  %36 = sext <8 x i16> %35 to <8 x i32>
  %37 = mul <8 x i32> %34, %36
  %38 = sub <8 x i32> %32, %37
  %39 = trunc <8 x i32> %38 to <8 x i16>
  call void @llvm.genx.vstore.v8i16.p0v8i16(<8 x i16> %39, <8 x i16>* %2)
  %40 = load <8 x i16>, <8 x i16>* %12, align 16, !tbaa !7
  ret <8 x i16> %40
}

; Function Attrs: noinline nounwind
define dso_local <16 x i16> @_Z24__cm_intrinsic_impl_sdivu2CMvb16_sS_(<16 x i16>, <16 x i16>) #11 {
  %3 = alloca <16 x i16>, align 32
  %4 = alloca <16 x i16>, align 32
  %5 = alloca <16 x i16>, align 32
  %6 = alloca <16 x i16>, align 32
  store <16 x i16> %0, <16 x i16>* %3, align 32, !tbaa !7
  store <16 x i16> %1, <16 x i16>* %4, align 32, !tbaa !7
  %7 = load <16 x i16>, <16 x i16>* %3, align 32, !tbaa !7
  %8 = load <16 x i16>, <16 x i16>* %4, align 32, !tbaa !7
  %9 = call <16 x i16> @_ZN7details13__impl_divremILi16EEEu2CMvbT__sS1_S1_u2CMvrT__s(<16 x i16> %7, <16 x i16> %8, <16 x i16>* %5)
  store <16 x i16> %9, <16 x i16>* %6, align 32, !tbaa !7
  %10 = load <16 x i16>, <16 x i16>* %6, align 32, !tbaa !7
  ret <16 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <16 x i16> @_ZN7details13__impl_divremILi16EEEu2CMvbT__sS1_S1_u2CMvrT__s(<16 x i16>, <16 x i16>, <16 x i16>*) #10 comdat {
  %4 = alloca <16 x i16>, align 32
  %5 = alloca <16 x i16>, align 32
  %6 = alloca <16 x i16>*, align 32
  %7 = alloca <16 x float>, align 64
  %8 = alloca <16 x float>, align 64
  %9 = alloca <16 x float>, align 64
  %10 = alloca <16 x float>, align 64
  %11 = alloca <16 x float>, align 64
  %12 = alloca <16 x i16>, align 32
  store <16 x i16> %0, <16 x i16>* %4, align 32, !tbaa !7
  store <16 x i16> %1, <16 x i16>* %5, align 32, !tbaa !7
  store <16 x i16>* %2, <16 x i16>** %6, align 32, !tbaa !7
  %13 = load <16 x i16>, <16 x i16>* %4, align 32, !tbaa !7
  %14 = sitofp <16 x i16> %13 to <16 x float>
  store <16 x float> %14, <16 x float>* %7, align 64, !tbaa !7
  %15 = load <16 x i16>, <16 x i16>* %5, align 32, !tbaa !7
  %16 = sitofp <16 x i16> %15 to <16 x float>
  store <16 x float> %16, <16 x float>* %8, align 64, !tbaa !7
  %17 = load <16 x float>, <16 x float>* %8, align 64, !tbaa !7
  %18 = fdiv <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %17
  store <16 x float> %18, <16 x float>* %9, align 64, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <16 x float> undef, float %19, i32 0
  %21 = shufflevector <16 x float> %20, <16 x float> undef, <16 x i32> zeroinitializer
  store <16 x float> %21, <16 x float>* %10, align 64, !tbaa !7
  %22 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %23 = load <16 x float>, <16 x float>* %10, align 64, !tbaa !7
  %24 = fmul <16 x float> %22, %23
  store <16 x float> %24, <16 x float>* %7, align 64, !tbaa !7
  %25 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %26 = load <16 x float>, <16 x float>* %9, align 64, !tbaa !7
  %27 = fmul <16 x float> %25, %26
  store <16 x float> %27, <16 x float>* %11, align 64, !tbaa !7
  %28 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %29 = fptosi <16 x float> %28 to <16 x i32>
  %30 = trunc <16 x i32> %29 to <16 x i16>
  store <16 x i16> %30, <16 x i16>* %12, align 32, !tbaa !7
  %31 = load <16 x i16>, <16 x i16>* %4, align 32, !tbaa !7
  %32 = sext <16 x i16> %31 to <16 x i32>
  %33 = load <16 x i16>, <16 x i16>* %12, align 32, !tbaa !7
  %34 = sext <16 x i16> %33 to <16 x i32>
  %35 = load <16 x i16>, <16 x i16>* %5, align 32, !tbaa !7
  %36 = sext <16 x i16> %35 to <16 x i32>
  %37 = mul <16 x i32> %34, %36
  %38 = sub <16 x i32> %32, %37
  %39 = trunc <16 x i32> %38 to <16 x i16>
  call void @llvm.genx.vstore.v16i16.p0v16i16(<16 x i16> %39, <16 x i16>* %2)
  %40 = load <16 x i16>, <16 x i16>* %12, align 32, !tbaa !7
  ret <16 x i16> %40
}

; Function Attrs: noinline nounwind
define internal <32 x i16> @_Z24__cm_intrinsic_impl_sdivu2CMvb32_sS_(<32 x i16>, <32 x i16>) #13 {
  %3 = alloca <32 x i16>, align 64
  %4 = alloca <32 x i16>, align 64
  %5 = alloca <32 x i16>, align 64
  %6 = alloca <32 x i16>, align 64
  store <32 x i16> %0, <32 x i16>* %3, align 64, !tbaa !7
  store <32 x i16> %1, <32 x i16>* %4, align 64, !tbaa !7
  %7 = load <32 x i16>, <32 x i16>* %3, align 64, !tbaa !7
  %8 = load <32 x i16>, <32 x i16>* %4, align 64, !tbaa !7
  %9 = call <32 x i16> @_ZN7details13__impl_divremILi32EEEu2CMvbT__sS1_S1_u2CMvrT__s(<32 x i16> %7, <32 x i16> %8, <32 x i16>* %5)
  store <32 x i16> %9, <32 x i16>* %6, align 64, !tbaa !7
  %10 = load <32 x i16>, <32 x i16>* %6, align 64, !tbaa !7
  ret <32 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <32 x i16> @_ZN7details13__impl_divremILi32EEEu2CMvbT__sS1_S1_u2CMvrT__s(<32 x i16>, <32 x i16>, <32 x i16>*) #12 comdat {
  %4 = alloca <32 x i16>, align 64
  %5 = alloca <32 x i16>, align 64
  %6 = alloca <32 x i16>*, align 64
  %7 = alloca <32 x float>, align 128
  %8 = alloca <32 x float>, align 128
  %9 = alloca <32 x float>, align 128
  %10 = alloca <32 x float>, align 128
  %11 = alloca <32 x float>, align 128
  %12 = alloca <32 x i16>, align 64
  store <32 x i16> %0, <32 x i16>* %4, align 64, !tbaa !7
  store <32 x i16> %1, <32 x i16>* %5, align 64, !tbaa !7
  store <32 x i16>* %2, <32 x i16>** %6, align 64, !tbaa !7
  %13 = load <32 x i16>, <32 x i16>* %4, align 64, !tbaa !7
  %14 = sitofp <32 x i16> %13 to <32 x float>
  store <32 x float> %14, <32 x float>* %7, align 128, !tbaa !7
  %15 = load <32 x i16>, <32 x i16>* %5, align 64, !tbaa !7
  %16 = sitofp <32 x i16> %15 to <32 x float>
  store <32 x float> %16, <32 x float>* %8, align 128, !tbaa !7
  %17 = load <32 x float>, <32 x float>* %8, align 128, !tbaa !7
  %18 = fdiv <32 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %17
  store <32 x float> %18, <32 x float>* %9, align 128, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <32 x float> undef, float %19, i32 0
  %21 = shufflevector <32 x float> %20, <32 x float> undef, <32 x i32> zeroinitializer
  store <32 x float> %21, <32 x float>* %10, align 128, !tbaa !7
  %22 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %23 = load <32 x float>, <32 x float>* %10, align 128, !tbaa !7
  %24 = fmul <32 x float> %22, %23
  store <32 x float> %24, <32 x float>* %7, align 128, !tbaa !7
  %25 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %26 = load <32 x float>, <32 x float>* %9, align 128, !tbaa !7
  %27 = fmul <32 x float> %25, %26
  store <32 x float> %27, <32 x float>* %11, align 128, !tbaa !7
  %28 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %29 = fptosi <32 x float> %28 to <32 x i32>
  %30 = trunc <32 x i32> %29 to <32 x i16>
  store <32 x i16> %30, <32 x i16>* %12, align 64, !tbaa !7
  %31 = load <32 x i16>, <32 x i16>* %4, align 64, !tbaa !7
  %32 = sext <32 x i16> %31 to <32 x i32>
  %33 = load <32 x i16>, <32 x i16>* %12, align 64, !tbaa !7
  %34 = sext <32 x i16> %33 to <32 x i32>
  %35 = load <32 x i16>, <32 x i16>* %5, align 64, !tbaa !7
  %36 = sext <32 x i16> %35 to <32 x i32>
  %37 = mul <32 x i32> %34, %36
  %38 = sub <32 x i32> %32, %37
  %39 = trunc <32 x i32> %38 to <32 x i16>
  call void @llvm.genx.vstore.v32i16.p0v32i16(<32 x i16> %39, <32 x i16>* %2)
  %40 = load <32 x i16>, <32 x i16>* %12, align 64, !tbaa !7
  ret <32 x i16> %40
}

; Function Attrs: noinline nounwind
define internal signext i16 @_Z24__cm_intrinsic_impl_sremss(i16 signext, i16 signext) #3 {
  %3 = alloca i16, align 2
  %4 = alloca i16, align 2
  %5 = alloca <1 x i16>, align 2
  %6 = alloca <1 x i16>, align 2
  %7 = alloca <1 x i16>, align 2
  store i16 %0, i16* %3, align 2, !tbaa !11
  store i16 %1, i16* %4, align 2, !tbaa !11
  %8 = load i16, i16* %3, align 2, !tbaa !11
  %9 = insertelement <1 x i16> undef, i16 %8, i32 0
  %10 = shufflevector <1 x i16> %9, <1 x i16> undef, <1 x i32> zeroinitializer
  store <1 x i16> %10, <1 x i16>* %5, align 2, !tbaa !7
  %11 = load i16, i16* %4, align 2, !tbaa !11
  %12 = insertelement <1 x i16> undef, i16 %11, i32 0
  %13 = shufflevector <1 x i16> %12, <1 x i16> undef, <1 x i32> zeroinitializer
  store <1 x i16> %13, <1 x i16>* %6, align 2, !tbaa !7
  %14 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  %15 = load <1 x i16>, <1 x i16>* %6, align 2, !tbaa !7
  %16 = call <1 x i16> @_ZN7details13__impl_divremILi1EEEu2CMvbT__sS1_S1_u2CMvrT__s(<1 x i16> %14, <1 x i16> %15, <1 x i16>* %7)
  %17 = load <1 x i16>, <1 x i16>* %7, align 2, !tbaa !7
  %18 = call <1 x i16> @llvm.genx.rdregioni.v1i16.v1i16.i16(<1 x i16> %17, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %19 = extractelement <1 x i16> %18, i32 0
  ret i16 %19
}

; Function Attrs: noinline nounwind
define internal <1 x i16> @_Z24__cm_intrinsic_impl_sremu2CMvb1_sS_(<1 x i16>, <1 x i16>) #3 {
  %3 = alloca <1 x i16>, align 2
  %4 = alloca <1 x i16>, align 2
  %5 = alloca <1 x i16>, align 2
  store <1 x i16> %0, <1 x i16>* %3, align 2, !tbaa !7
  store <1 x i16> %1, <1 x i16>* %4, align 2, !tbaa !7
  %6 = load <1 x i16>, <1 x i16>* %3, align 2, !tbaa !7
  %7 = load <1 x i16>, <1 x i16>* %4, align 2, !tbaa !7
  %8 = call <1 x i16> @_ZN7details13__impl_divremILi1EEEu2CMvbT__sS1_S1_u2CMvrT__s(<1 x i16> %6, <1 x i16> %7, <1 x i16>* %5)
  %9 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  ret <1 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <2 x i16> @_Z24__cm_intrinsic_impl_sremu2CMvb2_sS_(<2 x i16>, <2 x i16>) #5 {
  %3 = alloca <2 x i16>, align 4
  %4 = alloca <2 x i16>, align 4
  %5 = alloca <2 x i16>, align 4
  store <2 x i16> %0, <2 x i16>* %3, align 4, !tbaa !7
  store <2 x i16> %1, <2 x i16>* %4, align 4, !tbaa !7
  %6 = load <2 x i16>, <2 x i16>* %3, align 4, !tbaa !7
  %7 = load <2 x i16>, <2 x i16>* %4, align 4, !tbaa !7
  %8 = call <2 x i16> @_ZN7details13__impl_divremILi2EEEu2CMvbT__sS1_S1_u2CMvrT__s(<2 x i16> %6, <2 x i16> %7, <2 x i16>* %5)
  %9 = load <2 x i16>, <2 x i16>* %5, align 4, !tbaa !7
  ret <2 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <4 x i16> @_Z24__cm_intrinsic_impl_sremu2CMvb4_sS_(<4 x i16>, <4 x i16>) #7 {
  %3 = alloca <4 x i16>, align 8
  %4 = alloca <4 x i16>, align 8
  %5 = alloca <4 x i16>, align 8
  store <4 x i16> %0, <4 x i16>* %3, align 8, !tbaa !7
  store <4 x i16> %1, <4 x i16>* %4, align 8, !tbaa !7
  %6 = load <4 x i16>, <4 x i16>* %3, align 8, !tbaa !7
  %7 = load <4 x i16>, <4 x i16>* %4, align 8, !tbaa !7
  %8 = call <4 x i16> @_ZN7details13__impl_divremILi4EEEu2CMvbT__sS1_S1_u2CMvrT__s(<4 x i16> %6, <4 x i16> %7, <4 x i16>* %5)
  %9 = load <4 x i16>, <4 x i16>* %5, align 8, !tbaa !7
  ret <4 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <8 x i16> @_Z24__cm_intrinsic_impl_sremu2CMvb8_sS_(<8 x i16>, <8 x i16>) #9 {
  %3 = alloca <8 x i16>, align 16
  %4 = alloca <8 x i16>, align 16
  %5 = alloca <8 x i16>, align 16
  store <8 x i16> %0, <8 x i16>* %3, align 16, !tbaa !7
  store <8 x i16> %1, <8 x i16>* %4, align 16, !tbaa !7
  %6 = load <8 x i16>, <8 x i16>* %3, align 16, !tbaa !7
  %7 = load <8 x i16>, <8 x i16>* %4, align 16, !tbaa !7
  %8 = call <8 x i16> @_ZN7details13__impl_divremILi8EEEu2CMvbT__sS1_S1_u2CMvrT__s(<8 x i16> %6, <8 x i16> %7, <8 x i16>* %5)
  %9 = load <8 x i16>, <8 x i16>* %5, align 16, !tbaa !7
  ret <8 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <16 x i16> @_Z24__cm_intrinsic_impl_sremu2CMvb16_sS_(<16 x i16>, <16 x i16>) #11 {
  %3 = alloca <16 x i16>, align 32
  %4 = alloca <16 x i16>, align 32
  %5 = alloca <16 x i16>, align 32
  store <16 x i16> %0, <16 x i16>* %3, align 32, !tbaa !7
  store <16 x i16> %1, <16 x i16>* %4, align 32, !tbaa !7
  %6 = load <16 x i16>, <16 x i16>* %3, align 32, !tbaa !7
  %7 = load <16 x i16>, <16 x i16>* %4, align 32, !tbaa !7
  %8 = call <16 x i16> @_ZN7details13__impl_divremILi16EEEu2CMvbT__sS1_S1_u2CMvrT__s(<16 x i16> %6, <16 x i16> %7, <16 x i16>* %5)
  %9 = load <16 x i16>, <16 x i16>* %5, align 32, !tbaa !7
  ret <16 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <32 x i16> @_Z24__cm_intrinsic_impl_sremu2CMvb32_sS_(<32 x i16>, <32 x i16>) #13 {
  %3 = alloca <32 x i16>, align 64
  %4 = alloca <32 x i16>, align 64
  %5 = alloca <32 x i16>, align 64
  store <32 x i16> %0, <32 x i16>* %3, align 64, !tbaa !7
  store <32 x i16> %1, <32 x i16>* %4, align 64, !tbaa !7
  %6 = load <32 x i16>, <32 x i16>* %3, align 64, !tbaa !7
  %7 = load <32 x i16>, <32 x i16>* %4, align 64, !tbaa !7
  %8 = call <32 x i16> @_ZN7details13__impl_divremILi32EEEu2CMvbT__sS1_S1_u2CMvrT__s(<32 x i16> %6, <32 x i16> %7, <32 x i16>* %5)
  %9 = load <32 x i16>, <32 x i16>* %5, align 64, !tbaa !7
  ret <32 x i16> %9
}

; Function Attrs: noinline nounwind
define internal i32 @_Z24__cm_intrinsic_impl_sdivii(i32, i32) #14 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca <1 x i32>, align 4
  %6 = alloca <1 x i32>, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <1 x i32>, align 4
  store i32 %0, i32* %3, align 4, !tbaa !13
  store i32 %1, i32* %4, align 4, !tbaa !13
  %9 = load i32, i32* %3, align 4, !tbaa !13
  %10 = insertelement <1 x i32> undef, i32 %9, i32 0
  %11 = shufflevector <1 x i32> %10, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %11, <1 x i32>* %5, align 4, !tbaa !7
  %12 = load i32, i32* %4, align 4, !tbaa !13
  %13 = insertelement <1 x i32> undef, i32 %12, i32 0
  %14 = shufflevector <1 x i32> %13, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %14, <1 x i32>* %6, align 4, !tbaa !7
  %15 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %16 = load <1 x i32>, <1 x i32>* %6, align 4, !tbaa !7
  %17 = call <1 x i32> @_ZN7details13__impl_divremILi1EEEu2CMvbT__iS1_S1_u2CMvrT__j(<1 x i32> %15, <1 x i32> %16, <1 x i32>* %7)
  store <1 x i32> %17, <1 x i32>* %8, align 4, !tbaa !7
  %18 = load <1 x i32>, <1 x i32>* %8, align 4, !tbaa !7
  %19 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v1i32.i16(<1 x i32> %18, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %20 = extractelement <1 x i32> %19, i32 0
  ret i32 %20
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i32> @_ZN7details13__impl_divremILi1EEEu2CMvbT__iS1_S1_u2CMvrT__j(<1 x i32>, <1 x i32>, <1 x i32>*) #4 comdat {
  %4 = alloca <1 x i32>, align 4
  %5 = alloca <1 x i32>, align 4
  %6 = alloca <1 x i32>*, align 4
  %7 = alloca <1 x float>, align 4
  %8 = alloca <1 x float>, align 4
  %9 = alloca <1 x float>, align 4
  %10 = alloca <1 x float>, align 4
  %11 = alloca <1 x float>, align 4
  %12 = alloca <1 x float>, align 4
  %13 = alloca <1 x float>, align 4
  %14 = alloca <1 x float>, align 4
  %15 = alloca <1 x float>, align 4
  %16 = alloca <1 x i32>, align 4
  %17 = alloca <1 x i32>, align 4
  %18 = alloca <1 x i32>, align 4
  %19 = alloca <1 x i32>, align 4
  %20 = alloca <1 x i32>, align 4
  %21 = alloca <1 x i32>, align 4
  %22 = alloca <1 x i32>, align 4
  %23 = alloca <1 x i32>, align 4
  %24 = alloca <1 x i32>, align 4
  %25 = alloca <1 x i32>, align 4
  %26 = alloca <1 x i32>, align 4
  %27 = alloca <1 x i32>, align 4
  %28 = alloca <1 x i32>, align 4
  %29 = alloca <1 x i32>, align 4
  %30 = alloca <1 x float>, align 4
  store <1 x i32> %0, <1 x i32>* %4, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %5, align 4, !tbaa !7
  store <1 x i32>* %2, <1 x i32>** %6, align 4, !tbaa !7
  %31 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %32 = ashr <1 x i32> %31, <i32 31>
  store <1 x i32> %32, <1 x i32>* %27, align 4, !tbaa !7
  %33 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %34 = ashr <1 x i32> %33, <i32 31>
  store <1 x i32> %34, <1 x i32>* %28, align 4, !tbaa !7
  %35 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %36 = load <1 x i32>, <1 x i32>* %27, align 4, !tbaa !7
  %37 = add <1 x i32> %35, %36
  %38 = load <1 x i32>, <1 x i32>* %27, align 4, !tbaa !7
  %39 = xor <1 x i32> %37, %38
  store <1 x i32> %39, <1 x i32>* %24, align 4, !tbaa !7
  %40 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %41 = load <1 x i32>, <1 x i32>* %28, align 4, !tbaa !7
  %42 = add <1 x i32> %40, %41
  %43 = load <1 x i32>, <1 x i32>* %28, align 4, !tbaa !7
  %44 = xor <1 x i32> %42, %43
  store <1 x i32> %44, <1 x i32>* %25, align 4, !tbaa !7
  %45 = load <1 x i32>, <1 x i32>* %25, align 4, !tbaa !7
  %46 = uitofp <1 x i32> %45 to <1 x float>
  store <1 x float> %46, <1 x float>* %9, align 4, !tbaa !7
  %47 = load <1 x float>, <1 x float>* %9, align 4, !tbaa !7
  %48 = fptosi <1 x float> %47 to <1 x i32>
  store <1 x i32> %48, <1 x i32>* %17, align 4, !tbaa !7
  %49 = load <1 x i32>, <1 x i32>* %25, align 4, !tbaa !7
  %50 = load <1 x i32>, <1 x i32>* %17, align 4, !tbaa !7
  %51 = sub <1 x i32> %49, %50
  store <1 x i32> %51, <1 x i32>* %19, align 4, !tbaa !7
  %52 = load <1 x i32>, <1 x i32>* %19, align 4, !tbaa !7
  %53 = uitofp <1 x i32> %52 to <1 x float>
  store <1 x float> %53, <1 x float>* %10, align 4, !tbaa !7
  %54 = load <1 x i32>, <1 x i32>* %24, align 4, !tbaa !7
  %55 = uitofp <1 x i32> %54 to <1 x float>
  store <1 x float> %55, <1 x float>* %7, align 4, !tbaa !7
  %56 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %57 = fptosi <1 x float> %56 to <1 x i32>
  store <1 x i32> %57, <1 x i32>* %16, align 4, !tbaa !7
  %58 = load <1 x i32>, <1 x i32>* %24, align 4, !tbaa !7
  %59 = load <1 x i32>, <1 x i32>* %16, align 4, !tbaa !7
  %60 = sub <1 x i32> %58, %59
  store <1 x i32> %60, <1 x i32>* %18, align 4, !tbaa !7
  %61 = load <1 x i32>, <1 x i32>* %18, align 4, !tbaa !7
  %62 = uitofp <1 x i32> %61 to <1 x float>
  store <1 x float> %62, <1 x float>* %8, align 4, !tbaa !7
  %63 = load <1 x float>, <1 x float>* %9, align 4, !tbaa !7
  %64 = fdiv <1 x float> <float 1.000000e+00>, %63
  store <1 x float> %64, <1 x float>* %11, align 4, !tbaa !7
  %65 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %66 = insertelement <1 x float> undef, float %65, i32 0
  %67 = shufflevector <1 x float> %66, <1 x float> undef, <1 x i32> zeroinitializer
  store <1 x float> %67, <1 x float>* %30, align 4, !tbaa !7
  %68 = load <1 x float>, <1 x float>* %30, align 4, !tbaa !7
  %69 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %70 = fmul <1 x float> %68, %69
  %71 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %72 = fadd <1 x float> %71, %70
  store <1 x float> %72, <1 x float>* %11, align 4, !tbaa !7
  %73 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %74 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %75 = fmul <1 x float> %73, %74
  store <1 x float> %75, <1 x float>* %12, align 4, !tbaa !7
  %76 = load <1 x float>, <1 x float>* %12, align 4, !tbaa !7
  %77 = fptosi <1 x float> %76 to <1 x i32>
  store <1 x i32> %77, <1 x i32>* %20, align 4, !tbaa !7
  %78 = load <1 x i32>, <1 x i32>* %20, align 4, !tbaa !7
  %79 = uitofp <1 x i32> %78 to <1 x float>
  store <1 x float> %79, <1 x float>* %12, align 4, !tbaa !7
  %80 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %81 = load <1 x float>, <1 x float>* %9, align 4, !tbaa !7
  %82 = load <1 x float>, <1 x float>* %12, align 4, !tbaa !7
  %83 = fmul <1 x float> %81, %82
  %84 = fsub <1 x float> %80, %83
  store <1 x float> %84, <1 x float>* %14, align 4, !tbaa !7
  %85 = load <1 x float>, <1 x float>* %8, align 4, !tbaa !7
  %86 = load <1 x float>, <1 x float>* %10, align 4, !tbaa !7
  %87 = load <1 x float>, <1 x float>* %12, align 4, !tbaa !7
  %88 = fmul <1 x float> %86, %87
  %89 = fsub <1 x float> %85, %88
  store <1 x float> %89, <1 x float>* %15, align 4, !tbaa !7
  %90 = load <1 x float>, <1 x float>* %14, align 4, !tbaa !7
  %91 = load <1 x float>, <1 x float>* %15, align 4, !tbaa !7
  %92 = fadd <1 x float> %90, %91
  store <1 x float> %92, <1 x float>* %15, align 4, !tbaa !7
  %93 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %94 = load <1 x float>, <1 x float>* %15, align 4, !tbaa !7
  %95 = fmul <1 x float> %93, %94
  store <1 x float> %95, <1 x float>* %13, align 4, !tbaa !7
  %96 = load <1 x float>, <1 x float>* %13, align 4, !tbaa !7
  %97 = fptosi <1 x float> %96 to <1 x i32>
  store <1 x i32> %97, <1 x i32>* %21, align 4, !tbaa !7
  %98 = load <1 x i32>, <1 x i32>* %20, align 4, !tbaa !7
  %99 = load <1 x i32>, <1 x i32>* %21, align 4, !tbaa !7
  %100 = add <1 x i32> %98, %99
  store <1 x i32> %100, <1 x i32>* %22, align 4, !tbaa !7
  %101 = load <1 x i32>, <1 x i32>* %27, align 4, !tbaa !7
  %102 = load <1 x i32>, <1 x i32>* %28, align 4, !tbaa !7
  %103 = xor <1 x i32> %101, %102
  store <1 x i32> %103, <1 x i32>* %29, align 4, !tbaa !7
  %104 = load <1 x i32>, <1 x i32>* %24, align 4, !tbaa !7
  %105 = load <1 x i32>, <1 x i32>* %25, align 4, !tbaa !7
  %106 = load <1 x i32>, <1 x i32>* %22, align 4, !tbaa !7
  %107 = mul <1 x i32> %105, %106
  %108 = sub <1 x i32> %104, %107
  store <1 x i32> %108, <1 x i32>* %23, align 4, !tbaa !7
  %109 = load <1 x i32>, <1 x i32>* %23, align 4, !tbaa !7
  %110 = load <1 x i32>, <1 x i32>* %25, align 4, !tbaa !7
  %111 = icmp uge <1 x i32> %109, %110
  %112 = zext <1 x i1> %111 to <1 x i16>
  %113 = trunc <1 x i16> %112 to <1 x i1>
  %114 = select <1 x i1> %113, <1 x i32> <i32 -1>, <1 x i32> zeroinitializer
  store <1 x i32> %114, <1 x i32>* %26, align 4, !tbaa !7
  %115 = load <1 x i32>, <1 x i32>* %26, align 4, !tbaa !7
  %116 = and <1 x i32> %115, <i32 1>
  %117 = load <1 x i32>, <1 x i32>* %22, align 4, !tbaa !7
  %118 = add <1 x i32> %117, %116
  store <1 x i32> %118, <1 x i32>* %22, align 4, !tbaa !7
  %119 = load <1 x i32>, <1 x i32>* %25, align 4, !tbaa !7
  %120 = load <1 x i32>, <1 x i32>* %26, align 4, !tbaa !7
  %121 = and <1 x i32> %119, %120
  %122 = load <1 x i32>, <1 x i32>* %23, align 4, !tbaa !7
  %123 = sub <1 x i32> %122, %121
  store <1 x i32> %123, <1 x i32>* %23, align 4, !tbaa !7
  %124 = load <1 x i32>, <1 x i32>* %27, align 4, !tbaa !7
  %125 = load <1 x i32>, <1 x i32>* %23, align 4, !tbaa !7
  %126 = add <1 x i32> %124, %125
  %127 = load <1 x i32>, <1 x i32>* %27, align 4, !tbaa !7
  %128 = xor <1 x i32> %126, %127
  store <1 x i32> %128, <1 x i32>* %2
  %129 = load <1 x i32>, <1 x i32>* %29, align 4, !tbaa !7
  %130 = load <1 x i32>, <1 x i32>* %22, align 4, !tbaa !7
  %131 = add <1 x i32> %129, %130
  %132 = load <1 x i32>, <1 x i32>* %29, align 4, !tbaa !7
  %133 = xor <1 x i32> %131, %132
  store <1 x i32> %133, <1 x i32>* %22, align 4, !tbaa !7
  %134 = load <1 x i32>, <1 x i32>* %22, align 4, !tbaa !7
  ret <1 x i32> %134
}

; Function Attrs: nounwind readnone
declare <1 x i32> @llvm.genx.rdregioni.v1i32.v1i32.i16(<1 x i32>, i32, i32, i32, i16, i32) #2

; Function Attrs: noinline nounwind
define internal <1 x i32> @_Z24__cm_intrinsic_impl_sdivu2CMvb1_iS_(<1 x i32>, <1 x i32>) #14 {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x i32>, align 4
  %5 = alloca <1 x i32>, align 4
  %6 = alloca <1 x i32>, align 4
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %4, align 4, !tbaa !7
  %7 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %8 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %9 = call <1 x i32> @_ZN7details13__impl_divremILi1EEEu2CMvbT__iS1_S1_u2CMvrT__j(<1 x i32> %7, <1 x i32> %8, <1 x i32>* %5)
  store <1 x i32> %9, <1 x i32>* %6, align 4, !tbaa !7
  %10 = load <1 x i32>, <1 x i32>* %6, align 4, !tbaa !7
  ret <1 x i32> %10
}

; Function Attrs: noinline nounwind
define internal <2 x i32> @_Z24__cm_intrinsic_impl_sdivu2CMvb2_iS_(<2 x i32>, <2 x i32>) #15 {
  %3 = alloca <2 x i32>, align 8
  %4 = alloca <2 x i32>, align 8
  %5 = alloca <2 x i32>, align 8
  %6 = alloca <2 x i32>, align 8
  store <2 x i32> %0, <2 x i32>* %3, align 8, !tbaa !7
  store <2 x i32> %1, <2 x i32>* %4, align 8, !tbaa !7
  %7 = load <2 x i32>, <2 x i32>* %3, align 8, !tbaa !7
  %8 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %9 = call <2 x i32> @_ZN7details13__impl_divremILi2EEEu2CMvbT__iS1_S1_u2CMvrT__j(<2 x i32> %7, <2 x i32> %8, <2 x i32>* %5)
  store <2 x i32> %9, <2 x i32>* %6, align 8, !tbaa !7
  %10 = load <2 x i32>, <2 x i32>* %6, align 8, !tbaa !7
  ret <2 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <2 x i32> @_ZN7details13__impl_divremILi2EEEu2CMvbT__iS1_S1_u2CMvrT__j(<2 x i32>, <2 x i32>, <2 x i32>*) #6 comdat {
  %4 = alloca <2 x i32>, align 8
  %5 = alloca <2 x i32>, align 8
  %6 = alloca <2 x i32>*, align 8
  %7 = alloca <2 x float>, align 8
  %8 = alloca <2 x float>, align 8
  %9 = alloca <2 x float>, align 8
  %10 = alloca <2 x float>, align 8
  %11 = alloca <2 x float>, align 8
  %12 = alloca <2 x float>, align 8
  %13 = alloca <2 x float>, align 8
  %14 = alloca <2 x float>, align 8
  %15 = alloca <2 x float>, align 8
  %16 = alloca <2 x i32>, align 8
  %17 = alloca <2 x i32>, align 8
  %18 = alloca <2 x i32>, align 8
  %19 = alloca <2 x i32>, align 8
  %20 = alloca <2 x i32>, align 8
  %21 = alloca <2 x i32>, align 8
  %22 = alloca <2 x i32>, align 8
  %23 = alloca <2 x i32>, align 8
  %24 = alloca <2 x i32>, align 8
  %25 = alloca <2 x i32>, align 8
  %26 = alloca <2 x i32>, align 8
  %27 = alloca <2 x i32>, align 8
  %28 = alloca <2 x i32>, align 8
  %29 = alloca <2 x i32>, align 8
  %30 = alloca <2 x float>, align 8
  store <2 x i32> %0, <2 x i32>* %4, align 8, !tbaa !7
  store <2 x i32> %1, <2 x i32>* %5, align 8, !tbaa !7
  store <2 x i32>* %2, <2 x i32>** %6, align 8, !tbaa !7
  %31 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %32 = ashr <2 x i32> %31, <i32 31, i32 31>
  store <2 x i32> %32, <2 x i32>* %27, align 8, !tbaa !7
  %33 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  %34 = ashr <2 x i32> %33, <i32 31, i32 31>
  store <2 x i32> %34, <2 x i32>* %28, align 8, !tbaa !7
  %35 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %36 = load <2 x i32>, <2 x i32>* %27, align 8, !tbaa !7
  %37 = add <2 x i32> %35, %36
  %38 = load <2 x i32>, <2 x i32>* %27, align 8, !tbaa !7
  %39 = xor <2 x i32> %37, %38
  store <2 x i32> %39, <2 x i32>* %24, align 8, !tbaa !7
  %40 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  %41 = load <2 x i32>, <2 x i32>* %28, align 8, !tbaa !7
  %42 = add <2 x i32> %40, %41
  %43 = load <2 x i32>, <2 x i32>* %28, align 8, !tbaa !7
  %44 = xor <2 x i32> %42, %43
  store <2 x i32> %44, <2 x i32>* %25, align 8, !tbaa !7
  %45 = load <2 x i32>, <2 x i32>* %25, align 8, !tbaa !7
  %46 = uitofp <2 x i32> %45 to <2 x float>
  store <2 x float> %46, <2 x float>* %9, align 8, !tbaa !7
  %47 = load <2 x float>, <2 x float>* %9, align 8, !tbaa !7
  %48 = fptosi <2 x float> %47 to <2 x i32>
  store <2 x i32> %48, <2 x i32>* %17, align 8, !tbaa !7
  %49 = load <2 x i32>, <2 x i32>* %25, align 8, !tbaa !7
  %50 = load <2 x i32>, <2 x i32>* %17, align 8, !tbaa !7
  %51 = sub <2 x i32> %49, %50
  store <2 x i32> %51, <2 x i32>* %19, align 8, !tbaa !7
  %52 = load <2 x i32>, <2 x i32>* %19, align 8, !tbaa !7
  %53 = uitofp <2 x i32> %52 to <2 x float>
  store <2 x float> %53, <2 x float>* %10, align 8, !tbaa !7
  %54 = load <2 x i32>, <2 x i32>* %24, align 8, !tbaa !7
  %55 = uitofp <2 x i32> %54 to <2 x float>
  store <2 x float> %55, <2 x float>* %7, align 8, !tbaa !7
  %56 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %57 = fptosi <2 x float> %56 to <2 x i32>
  store <2 x i32> %57, <2 x i32>* %16, align 8, !tbaa !7
  %58 = load <2 x i32>, <2 x i32>* %24, align 8, !tbaa !7
  %59 = load <2 x i32>, <2 x i32>* %16, align 8, !tbaa !7
  %60 = sub <2 x i32> %58, %59
  store <2 x i32> %60, <2 x i32>* %18, align 8, !tbaa !7
  %61 = load <2 x i32>, <2 x i32>* %18, align 8, !tbaa !7
  %62 = uitofp <2 x i32> %61 to <2 x float>
  store <2 x float> %62, <2 x float>* %8, align 8, !tbaa !7
  %63 = load <2 x float>, <2 x float>* %9, align 8, !tbaa !7
  %64 = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %63
  store <2 x float> %64, <2 x float>* %11, align 8, !tbaa !7
  %65 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %66 = insertelement <2 x float> undef, float %65, i32 0
  %67 = shufflevector <2 x float> %66, <2 x float> undef, <2 x i32> zeroinitializer
  store <2 x float> %67, <2 x float>* %30, align 8, !tbaa !7
  %68 = load <2 x float>, <2 x float>* %30, align 8, !tbaa !7
  %69 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %70 = fmul <2 x float> %68, %69
  %71 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %72 = fadd <2 x float> %71, %70
  store <2 x float> %72, <2 x float>* %11, align 8, !tbaa !7
  %73 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %74 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %75 = fmul <2 x float> %73, %74
  store <2 x float> %75, <2 x float>* %12, align 8, !tbaa !7
  %76 = load <2 x float>, <2 x float>* %12, align 8, !tbaa !7
  %77 = fptosi <2 x float> %76 to <2 x i32>
  store <2 x i32> %77, <2 x i32>* %20, align 8, !tbaa !7
  %78 = load <2 x i32>, <2 x i32>* %20, align 8, !tbaa !7
  %79 = uitofp <2 x i32> %78 to <2 x float>
  store <2 x float> %79, <2 x float>* %12, align 8, !tbaa !7
  %80 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %81 = load <2 x float>, <2 x float>* %9, align 8, !tbaa !7
  %82 = load <2 x float>, <2 x float>* %12, align 8, !tbaa !7
  %83 = fmul <2 x float> %81, %82
  %84 = fsub <2 x float> %80, %83
  store <2 x float> %84, <2 x float>* %14, align 8, !tbaa !7
  %85 = load <2 x float>, <2 x float>* %8, align 8, !tbaa !7
  %86 = load <2 x float>, <2 x float>* %10, align 8, !tbaa !7
  %87 = load <2 x float>, <2 x float>* %12, align 8, !tbaa !7
  %88 = fmul <2 x float> %86, %87
  %89 = fsub <2 x float> %85, %88
  store <2 x float> %89, <2 x float>* %15, align 8, !tbaa !7
  %90 = load <2 x float>, <2 x float>* %14, align 8, !tbaa !7
  %91 = load <2 x float>, <2 x float>* %15, align 8, !tbaa !7
  %92 = fadd <2 x float> %90, %91
  store <2 x float> %92, <2 x float>* %15, align 8, !tbaa !7
  %93 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %94 = load <2 x float>, <2 x float>* %15, align 8, !tbaa !7
  %95 = fmul <2 x float> %93, %94
  store <2 x float> %95, <2 x float>* %13, align 8, !tbaa !7
  %96 = load <2 x float>, <2 x float>* %13, align 8, !tbaa !7
  %97 = fptosi <2 x float> %96 to <2 x i32>
  store <2 x i32> %97, <2 x i32>* %21, align 8, !tbaa !7
  %98 = load <2 x i32>, <2 x i32>* %20, align 8, !tbaa !7
  %99 = load <2 x i32>, <2 x i32>* %21, align 8, !tbaa !7
  %100 = add <2 x i32> %98, %99
  store <2 x i32> %100, <2 x i32>* %22, align 8, !tbaa !7
  %101 = load <2 x i32>, <2 x i32>* %27, align 8, !tbaa !7
  %102 = load <2 x i32>, <2 x i32>* %28, align 8, !tbaa !7
  %103 = xor <2 x i32> %101, %102
  store <2 x i32> %103, <2 x i32>* %29, align 8, !tbaa !7
  %104 = load <2 x i32>, <2 x i32>* %24, align 8, !tbaa !7
  %105 = load <2 x i32>, <2 x i32>* %25, align 8, !tbaa !7
  %106 = load <2 x i32>, <2 x i32>* %22, align 8, !tbaa !7
  %107 = mul <2 x i32> %105, %106
  %108 = sub <2 x i32> %104, %107
  store <2 x i32> %108, <2 x i32>* %23, align 8, !tbaa !7
  %109 = load <2 x i32>, <2 x i32>* %23, align 8, !tbaa !7
  %110 = load <2 x i32>, <2 x i32>* %25, align 8, !tbaa !7
  %111 = icmp uge <2 x i32> %109, %110
  %112 = zext <2 x i1> %111 to <2 x i16>
  %113 = trunc <2 x i16> %112 to <2 x i1>
  %114 = select <2 x i1> %113, <2 x i32> <i32 -1, i32 -1>, <2 x i32> zeroinitializer
  store <2 x i32> %114, <2 x i32>* %26, align 8, !tbaa !7
  %115 = load <2 x i32>, <2 x i32>* %26, align 8, !tbaa !7
  %116 = and <2 x i32> %115, <i32 1, i32 1>
  %117 = load <2 x i32>, <2 x i32>* %22, align 8, !tbaa !7
  %118 = add <2 x i32> %117, %116
  store <2 x i32> %118, <2 x i32>* %22, align 8, !tbaa !7
  %119 = load <2 x i32>, <2 x i32>* %25, align 8, !tbaa !7
  %120 = load <2 x i32>, <2 x i32>* %26, align 8, !tbaa !7
  %121 = and <2 x i32> %119, %120
  %122 = load <2 x i32>, <2 x i32>* %23, align 8, !tbaa !7
  %123 = sub <2 x i32> %122, %121
  store <2 x i32> %123, <2 x i32>* %23, align 8, !tbaa !7
  %124 = load <2 x i32>, <2 x i32>* %27, align 8, !tbaa !7
  %125 = load <2 x i32>, <2 x i32>* %23, align 8, !tbaa !7
  %126 = add <2 x i32> %124, %125
  %127 = load <2 x i32>, <2 x i32>* %27, align 8, !tbaa !7
  %128 = xor <2 x i32> %126, %127
  call void @llvm.genx.vstore.v2i32.p0v2i32(<2 x i32> %128, <2 x i32>* %2)
  %129 = load <2 x i32>, <2 x i32>* %29, align 8, !tbaa !7
  %130 = load <2 x i32>, <2 x i32>* %22, align 8, !tbaa !7
  %131 = add <2 x i32> %129, %130
  %132 = load <2 x i32>, <2 x i32>* %29, align 8, !tbaa !7
  %133 = xor <2 x i32> %131, %132
  store <2 x i32> %133, <2 x i32>* %22, align 8, !tbaa !7
  %134 = load <2 x i32>, <2 x i32>* %22, align 8, !tbaa !7
  ret <2 x i32> %134
}

; Function Attrs: noinline nounwind
define internal <4 x i32> @_Z24__cm_intrinsic_impl_sdivu2CMvb4_iS_(<4 x i32>, <4 x i32>) #16 {
  %3 = alloca <4 x i32>, align 16
  %4 = alloca <4 x i32>, align 16
  %5 = alloca <4 x i32>, align 16
  %6 = alloca <4 x i32>, align 16
  store <4 x i32> %0, <4 x i32>* %3, align 16, !tbaa !7
  store <4 x i32> %1, <4 x i32>* %4, align 16, !tbaa !7
  %7 = load <4 x i32>, <4 x i32>* %3, align 16, !tbaa !7
  %8 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %9 = call <4 x i32> @_ZN7details13__impl_divremILi4EEEu2CMvbT__iS1_S1_u2CMvrT__j(<4 x i32> %7, <4 x i32> %8, <4 x i32>* %5)
  store <4 x i32> %9, <4 x i32>* %6, align 16, !tbaa !7
  %10 = load <4 x i32>, <4 x i32>* %6, align 16, !tbaa !7
  ret <4 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <4 x i32> @_ZN7details13__impl_divremILi4EEEu2CMvbT__iS1_S1_u2CMvrT__j(<4 x i32>, <4 x i32>, <4 x i32>*) #8 comdat {
  %4 = alloca <4 x i32>, align 16
  %5 = alloca <4 x i32>, align 16
  %6 = alloca <4 x i32>*, align 16
  %7 = alloca <4 x float>, align 16
  %8 = alloca <4 x float>, align 16
  %9 = alloca <4 x float>, align 16
  %10 = alloca <4 x float>, align 16
  %11 = alloca <4 x float>, align 16
  %12 = alloca <4 x float>, align 16
  %13 = alloca <4 x float>, align 16
  %14 = alloca <4 x float>, align 16
  %15 = alloca <4 x float>, align 16
  %16 = alloca <4 x i32>, align 16
  %17 = alloca <4 x i32>, align 16
  %18 = alloca <4 x i32>, align 16
  %19 = alloca <4 x i32>, align 16
  %20 = alloca <4 x i32>, align 16
  %21 = alloca <4 x i32>, align 16
  %22 = alloca <4 x i32>, align 16
  %23 = alloca <4 x i32>, align 16
  %24 = alloca <4 x i32>, align 16
  %25 = alloca <4 x i32>, align 16
  %26 = alloca <4 x i32>, align 16
  %27 = alloca <4 x i32>, align 16
  %28 = alloca <4 x i32>, align 16
  %29 = alloca <4 x i32>, align 16
  %30 = alloca <4 x float>, align 16
  store <4 x i32> %0, <4 x i32>* %4, align 16, !tbaa !7
  store <4 x i32> %1, <4 x i32>* %5, align 16, !tbaa !7
  store <4 x i32>* %2, <4 x i32>** %6, align 16, !tbaa !7
  %31 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %32 = ashr <4 x i32> %31, <i32 31, i32 31, i32 31, i32 31>
  store <4 x i32> %32, <4 x i32>* %27, align 16, !tbaa !7
  %33 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  %34 = ashr <4 x i32> %33, <i32 31, i32 31, i32 31, i32 31>
  store <4 x i32> %34, <4 x i32>* %28, align 16, !tbaa !7
  %35 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %36 = load <4 x i32>, <4 x i32>* %27, align 16, !tbaa !7
  %37 = add <4 x i32> %35, %36
  %38 = load <4 x i32>, <4 x i32>* %27, align 16, !tbaa !7
  %39 = xor <4 x i32> %37, %38
  store <4 x i32> %39, <4 x i32>* %24, align 16, !tbaa !7
  %40 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  %41 = load <4 x i32>, <4 x i32>* %28, align 16, !tbaa !7
  %42 = add <4 x i32> %40, %41
  %43 = load <4 x i32>, <4 x i32>* %28, align 16, !tbaa !7
  %44 = xor <4 x i32> %42, %43
  store <4 x i32> %44, <4 x i32>* %25, align 16, !tbaa !7
  %45 = load <4 x i32>, <4 x i32>* %25, align 16, !tbaa !7
  %46 = uitofp <4 x i32> %45 to <4 x float>
  store <4 x float> %46, <4 x float>* %9, align 16, !tbaa !7
  %47 = load <4 x float>, <4 x float>* %9, align 16, !tbaa !7
  %48 = fptosi <4 x float> %47 to <4 x i32>
  store <4 x i32> %48, <4 x i32>* %17, align 16, !tbaa !7
  %49 = load <4 x i32>, <4 x i32>* %25, align 16, !tbaa !7
  %50 = load <4 x i32>, <4 x i32>* %17, align 16, !tbaa !7
  %51 = sub <4 x i32> %49, %50
  store <4 x i32> %51, <4 x i32>* %19, align 16, !tbaa !7
  %52 = load <4 x i32>, <4 x i32>* %19, align 16, !tbaa !7
  %53 = uitofp <4 x i32> %52 to <4 x float>
  store <4 x float> %53, <4 x float>* %10, align 16, !tbaa !7
  %54 = load <4 x i32>, <4 x i32>* %24, align 16, !tbaa !7
  %55 = uitofp <4 x i32> %54 to <4 x float>
  store <4 x float> %55, <4 x float>* %7, align 16, !tbaa !7
  %56 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %57 = fptosi <4 x float> %56 to <4 x i32>
  store <4 x i32> %57, <4 x i32>* %16, align 16, !tbaa !7
  %58 = load <4 x i32>, <4 x i32>* %24, align 16, !tbaa !7
  %59 = load <4 x i32>, <4 x i32>* %16, align 16, !tbaa !7
  %60 = sub <4 x i32> %58, %59
  store <4 x i32> %60, <4 x i32>* %18, align 16, !tbaa !7
  %61 = load <4 x i32>, <4 x i32>* %18, align 16, !tbaa !7
  %62 = uitofp <4 x i32> %61 to <4 x float>
  store <4 x float> %62, <4 x float>* %8, align 16, !tbaa !7
  %63 = load <4 x float>, <4 x float>* %9, align 16, !tbaa !7
  %64 = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %63
  store <4 x float> %64, <4 x float>* %11, align 16, !tbaa !7
  %65 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %66 = insertelement <4 x float> undef, float %65, i32 0
  %67 = shufflevector <4 x float> %66, <4 x float> undef, <4 x i32> zeroinitializer
  store <4 x float> %67, <4 x float>* %30, align 16, !tbaa !7
  %68 = load <4 x float>, <4 x float>* %30, align 16, !tbaa !7
  %69 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %70 = fmul <4 x float> %68, %69
  %71 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %72 = fadd <4 x float> %71, %70
  store <4 x float> %72, <4 x float>* %11, align 16, !tbaa !7
  %73 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %74 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %75 = fmul <4 x float> %73, %74
  store <4 x float> %75, <4 x float>* %12, align 16, !tbaa !7
  %76 = load <4 x float>, <4 x float>* %12, align 16, !tbaa !7
  %77 = fptosi <4 x float> %76 to <4 x i32>
  store <4 x i32> %77, <4 x i32>* %20, align 16, !tbaa !7
  %78 = load <4 x i32>, <4 x i32>* %20, align 16, !tbaa !7
  %79 = uitofp <4 x i32> %78 to <4 x float>
  store <4 x float> %79, <4 x float>* %12, align 16, !tbaa !7
  %80 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %81 = load <4 x float>, <4 x float>* %9, align 16, !tbaa !7
  %82 = load <4 x float>, <4 x float>* %12, align 16, !tbaa !7
  %83 = fmul <4 x float> %81, %82
  %84 = fsub <4 x float> %80, %83
  store <4 x float> %84, <4 x float>* %14, align 16, !tbaa !7
  %85 = load <4 x float>, <4 x float>* %8, align 16, !tbaa !7
  %86 = load <4 x float>, <4 x float>* %10, align 16, !tbaa !7
  %87 = load <4 x float>, <4 x float>* %12, align 16, !tbaa !7
  %88 = fmul <4 x float> %86, %87
  %89 = fsub <4 x float> %85, %88
  store <4 x float> %89, <4 x float>* %15, align 16, !tbaa !7
  %90 = load <4 x float>, <4 x float>* %14, align 16, !tbaa !7
  %91 = load <4 x float>, <4 x float>* %15, align 16, !tbaa !7
  %92 = fadd <4 x float> %90, %91
  store <4 x float> %92, <4 x float>* %15, align 16, !tbaa !7
  %93 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %94 = load <4 x float>, <4 x float>* %15, align 16, !tbaa !7
  %95 = fmul <4 x float> %93, %94
  store <4 x float> %95, <4 x float>* %13, align 16, !tbaa !7
  %96 = load <4 x float>, <4 x float>* %13, align 16, !tbaa !7
  %97 = fptosi <4 x float> %96 to <4 x i32>
  store <4 x i32> %97, <4 x i32>* %21, align 16, !tbaa !7
  %98 = load <4 x i32>, <4 x i32>* %20, align 16, !tbaa !7
  %99 = load <4 x i32>, <4 x i32>* %21, align 16, !tbaa !7
  %100 = add <4 x i32> %98, %99
  store <4 x i32> %100, <4 x i32>* %22, align 16, !tbaa !7
  %101 = load <4 x i32>, <4 x i32>* %27, align 16, !tbaa !7
  %102 = load <4 x i32>, <4 x i32>* %28, align 16, !tbaa !7
  %103 = xor <4 x i32> %101, %102
  store <4 x i32> %103, <4 x i32>* %29, align 16, !tbaa !7
  %104 = load <4 x i32>, <4 x i32>* %24, align 16, !tbaa !7
  %105 = load <4 x i32>, <4 x i32>* %25, align 16, !tbaa !7
  %106 = load <4 x i32>, <4 x i32>* %22, align 16, !tbaa !7
  %107 = mul <4 x i32> %105, %106
  %108 = sub <4 x i32> %104, %107
  store <4 x i32> %108, <4 x i32>* %23, align 16, !tbaa !7
  %109 = load <4 x i32>, <4 x i32>* %23, align 16, !tbaa !7
  %110 = load <4 x i32>, <4 x i32>* %25, align 16, !tbaa !7
  %111 = icmp uge <4 x i32> %109, %110
  %112 = zext <4 x i1> %111 to <4 x i16>
  %113 = trunc <4 x i16> %112 to <4 x i1>
  %114 = select <4 x i1> %113, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer
  store <4 x i32> %114, <4 x i32>* %26, align 16, !tbaa !7
  %115 = load <4 x i32>, <4 x i32>* %26, align 16, !tbaa !7
  %116 = and <4 x i32> %115, <i32 1, i32 1, i32 1, i32 1>
  %117 = load <4 x i32>, <4 x i32>* %22, align 16, !tbaa !7
  %118 = add <4 x i32> %117, %116
  store <4 x i32> %118, <4 x i32>* %22, align 16, !tbaa !7
  %119 = load <4 x i32>, <4 x i32>* %25, align 16, !tbaa !7
  %120 = load <4 x i32>, <4 x i32>* %26, align 16, !tbaa !7
  %121 = and <4 x i32> %119, %120
  %122 = load <4 x i32>, <4 x i32>* %23, align 16, !tbaa !7
  %123 = sub <4 x i32> %122, %121
  store <4 x i32> %123, <4 x i32>* %23, align 16, !tbaa !7
  %124 = load <4 x i32>, <4 x i32>* %27, align 16, !tbaa !7
  %125 = load <4 x i32>, <4 x i32>* %23, align 16, !tbaa !7
  %126 = add <4 x i32> %124, %125
  %127 = load <4 x i32>, <4 x i32>* %27, align 16, !tbaa !7
  %128 = xor <4 x i32> %126, %127
  call void @llvm.genx.vstore.v4i32.p0v4i32(<4 x i32> %128, <4 x i32>* %2)
  %129 = load <4 x i32>, <4 x i32>* %29, align 16, !tbaa !7
  %130 = load <4 x i32>, <4 x i32>* %22, align 16, !tbaa !7
  %131 = add <4 x i32> %129, %130
  %132 = load <4 x i32>, <4 x i32>* %29, align 16, !tbaa !7
  %133 = xor <4 x i32> %131, %132
  store <4 x i32> %133, <4 x i32>* %22, align 16, !tbaa !7
  %134 = load <4 x i32>, <4 x i32>* %22, align 16, !tbaa !7
  ret <4 x i32> %134
}

; Function Attrs: noinline nounwind
define dso_local <8 x i32> @_Z24__cm_intrinsic_impl_sdivu2CMvb8_iS_(<8 x i32>, <8 x i32>) #17 {
  %3 = alloca <8 x i32>, align 32
  %4 = alloca <8 x i32>, align 32
  %5 = alloca <8 x i32>, align 32
  %6 = alloca <8 x i32>, align 32
  store <8 x i32> %0, <8 x i32>* %3, align 32, !tbaa !7
  store <8 x i32> %1, <8 x i32>* %4, align 32, !tbaa !7
  %7 = load <8 x i32>, <8 x i32>* %3, align 32, !tbaa !7
  %8 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %9 = call <8 x i32> @_ZN7details13__impl_divremILi8EEEu2CMvbT__iS1_S1_u2CMvrT__j(<8 x i32> %7, <8 x i32> %8, <8 x i32>* %5)
  store <8 x i32> %9, <8 x i32>* %6, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %6, align 32, !tbaa !7
  ret <8 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <8 x i32> @_ZN7details13__impl_divremILi8EEEu2CMvbT__iS1_S1_u2CMvrT__j(<8 x i32>, <8 x i32>, <8 x i32>*) #10 comdat {
  %4 = alloca <8 x i32>, align 32
  %5 = alloca <8 x i32>, align 32
  %6 = alloca <8 x i32>*, align 32
  %7 = alloca <8 x float>, align 32
  %8 = alloca <8 x float>, align 32
  %9 = alloca <8 x float>, align 32
  %10 = alloca <8 x float>, align 32
  %11 = alloca <8 x float>, align 32
  %12 = alloca <8 x float>, align 32
  %13 = alloca <8 x float>, align 32
  %14 = alloca <8 x float>, align 32
  %15 = alloca <8 x float>, align 32
  %16 = alloca <8 x i32>, align 32
  %17 = alloca <8 x i32>, align 32
  %18 = alloca <8 x i32>, align 32
  %19 = alloca <8 x i32>, align 32
  %20 = alloca <8 x i32>, align 32
  %21 = alloca <8 x i32>, align 32
  %22 = alloca <8 x i32>, align 32
  %23 = alloca <8 x i32>, align 32
  %24 = alloca <8 x i32>, align 32
  %25 = alloca <8 x i32>, align 32
  %26 = alloca <8 x i32>, align 32
  %27 = alloca <8 x i32>, align 32
  %28 = alloca <8 x i32>, align 32
  %29 = alloca <8 x i32>, align 32
  %30 = alloca <8 x float>, align 32
  store <8 x i32> %0, <8 x i32>* %4, align 32, !tbaa !7
  store <8 x i32> %1, <8 x i32>* %5, align 32, !tbaa !7
  store <8 x i32>* %2, <8 x i32>** %6, align 32, !tbaa !7
  %31 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %32 = ashr <8 x i32> %31, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  store <8 x i32> %32, <8 x i32>* %27, align 32, !tbaa !7
  %33 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %34 = ashr <8 x i32> %33, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  store <8 x i32> %34, <8 x i32>* %28, align 32, !tbaa !7
  %35 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %36 = load <8 x i32>, <8 x i32>* %27, align 32, !tbaa !7
  %37 = add <8 x i32> %35, %36
  %38 = load <8 x i32>, <8 x i32>* %27, align 32, !tbaa !7
  %39 = xor <8 x i32> %37, %38
  store <8 x i32> %39, <8 x i32>* %24, align 32, !tbaa !7
  %40 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %41 = load <8 x i32>, <8 x i32>* %28, align 32, !tbaa !7
  %42 = add <8 x i32> %40, %41
  %43 = load <8 x i32>, <8 x i32>* %28, align 32, !tbaa !7
  %44 = xor <8 x i32> %42, %43
  store <8 x i32> %44, <8 x i32>* %25, align 32, !tbaa !7
  %45 = load <8 x i32>, <8 x i32>* %25, align 32, !tbaa !7
  %46 = uitofp <8 x i32> %45 to <8 x float>
  store <8 x float> %46, <8 x float>* %9, align 32, !tbaa !7
  %47 = load <8 x float>, <8 x float>* %9, align 32, !tbaa !7
  %48 = fptosi <8 x float> %47 to <8 x i32>
  store <8 x i32> %48, <8 x i32>* %17, align 32, !tbaa !7
  %49 = load <8 x i32>, <8 x i32>* %25, align 32, !tbaa !7
  %50 = load <8 x i32>, <8 x i32>* %17, align 32, !tbaa !7
  %51 = sub <8 x i32> %49, %50
  store <8 x i32> %51, <8 x i32>* %19, align 32, !tbaa !7
  %52 = load <8 x i32>, <8 x i32>* %19, align 32, !tbaa !7
  %53 = uitofp <8 x i32> %52 to <8 x float>
  store <8 x float> %53, <8 x float>* %10, align 32, !tbaa !7
  %54 = load <8 x i32>, <8 x i32>* %24, align 32, !tbaa !7
  %55 = uitofp <8 x i32> %54 to <8 x float>
  store <8 x float> %55, <8 x float>* %7, align 32, !tbaa !7
  %56 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %57 = fptosi <8 x float> %56 to <8 x i32>
  store <8 x i32> %57, <8 x i32>* %16, align 32, !tbaa !7
  %58 = load <8 x i32>, <8 x i32>* %24, align 32, !tbaa !7
  %59 = load <8 x i32>, <8 x i32>* %16, align 32, !tbaa !7
  %60 = sub <8 x i32> %58, %59
  store <8 x i32> %60, <8 x i32>* %18, align 32, !tbaa !7
  %61 = load <8 x i32>, <8 x i32>* %18, align 32, !tbaa !7
  %62 = uitofp <8 x i32> %61 to <8 x float>
  store <8 x float> %62, <8 x float>* %8, align 32, !tbaa !7
  %63 = load <8 x float>, <8 x float>* %9, align 32, !tbaa !7
  %64 = fdiv <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %63
  store <8 x float> %64, <8 x float>* %11, align 32, !tbaa !7
  %65 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %66 = insertelement <8 x float> undef, float %65, i32 0
  %67 = shufflevector <8 x float> %66, <8 x float> undef, <8 x i32> zeroinitializer
  store <8 x float> %67, <8 x float>* %30, align 32, !tbaa !7
  %68 = load <8 x float>, <8 x float>* %30, align 32, !tbaa !7
  %69 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %70 = fmul <8 x float> %68, %69
  %71 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %72 = fadd <8 x float> %71, %70
  store <8 x float> %72, <8 x float>* %11, align 32, !tbaa !7
  %73 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %74 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %75 = fmul <8 x float> %73, %74
  store <8 x float> %75, <8 x float>* %12, align 32, !tbaa !7
  %76 = load <8 x float>, <8 x float>* %12, align 32, !tbaa !7
  %77 = fptosi <8 x float> %76 to <8 x i32>
  store <8 x i32> %77, <8 x i32>* %20, align 32, !tbaa !7
  %78 = load <8 x i32>, <8 x i32>* %20, align 32, !tbaa !7
  %79 = uitofp <8 x i32> %78 to <8 x float>
  store <8 x float> %79, <8 x float>* %12, align 32, !tbaa !7
  %80 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %81 = load <8 x float>, <8 x float>* %9, align 32, !tbaa !7
  %82 = load <8 x float>, <8 x float>* %12, align 32, !tbaa !7
  %83 = fmul <8 x float> %81, %82
  %84 = fsub <8 x float> %80, %83
  store <8 x float> %84, <8 x float>* %14, align 32, !tbaa !7
  %85 = load <8 x float>, <8 x float>* %8, align 32, !tbaa !7
  %86 = load <8 x float>, <8 x float>* %10, align 32, !tbaa !7
  %87 = load <8 x float>, <8 x float>* %12, align 32, !tbaa !7
  %88 = fmul <8 x float> %86, %87
  %89 = fsub <8 x float> %85, %88
  store <8 x float> %89, <8 x float>* %15, align 32, !tbaa !7
  %90 = load <8 x float>, <8 x float>* %14, align 32, !tbaa !7
  %91 = load <8 x float>, <8 x float>* %15, align 32, !tbaa !7
  %92 = fadd <8 x float> %90, %91
  store <8 x float> %92, <8 x float>* %15, align 32, !tbaa !7
  %93 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %94 = load <8 x float>, <8 x float>* %15, align 32, !tbaa !7
  %95 = fmul <8 x float> %93, %94
  store <8 x float> %95, <8 x float>* %13, align 32, !tbaa !7
  %96 = load <8 x float>, <8 x float>* %13, align 32, !tbaa !7
  %97 = fptosi <8 x float> %96 to <8 x i32>
  store <8 x i32> %97, <8 x i32>* %21, align 32, !tbaa !7
  %98 = load <8 x i32>, <8 x i32>* %20, align 32, !tbaa !7
  %99 = load <8 x i32>, <8 x i32>* %21, align 32, !tbaa !7
  %100 = add <8 x i32> %98, %99
  store <8 x i32> %100, <8 x i32>* %22, align 32, !tbaa !7
  %101 = load <8 x i32>, <8 x i32>* %27, align 32, !tbaa !7
  %102 = load <8 x i32>, <8 x i32>* %28, align 32, !tbaa !7
  %103 = xor <8 x i32> %101, %102
  store <8 x i32> %103, <8 x i32>* %29, align 32, !tbaa !7
  %104 = load <8 x i32>, <8 x i32>* %24, align 32, !tbaa !7
  %105 = load <8 x i32>, <8 x i32>* %25, align 32, !tbaa !7
  %106 = load <8 x i32>, <8 x i32>* %22, align 32, !tbaa !7
  %107 = mul <8 x i32> %105, %106
  %108 = sub <8 x i32> %104, %107
  store <8 x i32> %108, <8 x i32>* %23, align 32, !tbaa !7
  %109 = load <8 x i32>, <8 x i32>* %23, align 32, !tbaa !7
  %110 = load <8 x i32>, <8 x i32>* %25, align 32, !tbaa !7
  %111 = icmp uge <8 x i32> %109, %110
  %112 = zext <8 x i1> %111 to <8 x i16>
  %113 = trunc <8 x i16> %112 to <8 x i1>
  %114 = select <8 x i1> %113, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer
  store <8 x i32> %114, <8 x i32>* %26, align 32, !tbaa !7
  %115 = load <8 x i32>, <8 x i32>* %26, align 32, !tbaa !7
  %116 = and <8 x i32> %115, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %117 = load <8 x i32>, <8 x i32>* %22, align 32, !tbaa !7
  %118 = add <8 x i32> %117, %116
  store <8 x i32> %118, <8 x i32>* %22, align 32, !tbaa !7
  %119 = load <8 x i32>, <8 x i32>* %25, align 32, !tbaa !7
  %120 = load <8 x i32>, <8 x i32>* %26, align 32, !tbaa !7
  %121 = and <8 x i32> %119, %120
  %122 = load <8 x i32>, <8 x i32>* %23, align 32, !tbaa !7
  %123 = sub <8 x i32> %122, %121
  store <8 x i32> %123, <8 x i32>* %23, align 32, !tbaa !7
  %124 = load <8 x i32>, <8 x i32>* %27, align 32, !tbaa !7
  %125 = load <8 x i32>, <8 x i32>* %23, align 32, !tbaa !7
  %126 = add <8 x i32> %124, %125
  %127 = load <8 x i32>, <8 x i32>* %27, align 32, !tbaa !7
  %128 = xor <8 x i32> %126, %127
  call void @llvm.genx.vstore.v8i32.p0v8i32(<8 x i32> %128, <8 x i32>* %2)
  %129 = load <8 x i32>, <8 x i32>* %29, align 32, !tbaa !7
  %130 = load <8 x i32>, <8 x i32>* %22, align 32, !tbaa !7
  %131 = add <8 x i32> %129, %130
  %132 = load <8 x i32>, <8 x i32>* %29, align 32, !tbaa !7
  %133 = xor <8 x i32> %131, %132
  store <8 x i32> %133, <8 x i32>* %22, align 32, !tbaa !7
  %134 = load <8 x i32>, <8 x i32>* %22, align 32, !tbaa !7
  ret <8 x i32> %134
}

; Function Attrs: noinline nounwind
define dso_local <16 x i32> @_Z24__cm_intrinsic_impl_sdivu2CMvb16_iS_(<16 x i32>, <16 x i32>) #18 {
  %3 = alloca <16 x i32>, align 64
  %4 = alloca <16 x i32>, align 64
  %5 = alloca <16 x i32>, align 64
  %6 = alloca <16 x i32>, align 64
  store <16 x i32> %0, <16 x i32>* %3, align 64, !tbaa !7
  store <16 x i32> %1, <16 x i32>* %4, align 64, !tbaa !7
  %7 = load <16 x i32>, <16 x i32>* %3, align 64, !tbaa !7
  %8 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %9 = call <16 x i32> @_ZN7details13__impl_divremILi16EEEu2CMvbT__iS1_S1_u2CMvrT__j(<16 x i32> %7, <16 x i32> %8, <16 x i32>* %5)
  store <16 x i32> %9, <16 x i32>* %6, align 64, !tbaa !7
  %10 = load <16 x i32>, <16 x i32>* %6, align 64, !tbaa !7
  ret <16 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <16 x i32> @_ZN7details13__impl_divremILi16EEEu2CMvbT__iS1_S1_u2CMvrT__j(<16 x i32>, <16 x i32>, <16 x i32>*) #12 comdat {
  %4 = alloca <16 x i32>, align 64
  %5 = alloca <16 x i32>, align 64
  %6 = alloca <16 x i32>*, align 64
  %7 = alloca <16 x float>, align 64
  %8 = alloca <16 x float>, align 64
  %9 = alloca <16 x float>, align 64
  %10 = alloca <16 x float>, align 64
  %11 = alloca <16 x float>, align 64
  %12 = alloca <16 x float>, align 64
  %13 = alloca <16 x float>, align 64
  %14 = alloca <16 x float>, align 64
  %15 = alloca <16 x float>, align 64
  %16 = alloca <16 x i32>, align 64
  %17 = alloca <16 x i32>, align 64
  %18 = alloca <16 x i32>, align 64
  %19 = alloca <16 x i32>, align 64
  %20 = alloca <16 x i32>, align 64
  %21 = alloca <16 x i32>, align 64
  %22 = alloca <16 x i32>, align 64
  %23 = alloca <16 x i32>, align 64
  %24 = alloca <16 x i32>, align 64
  %25 = alloca <16 x i32>, align 64
  %26 = alloca <16 x i32>, align 64
  %27 = alloca <16 x i32>, align 64
  %28 = alloca <16 x i32>, align 64
  %29 = alloca <16 x i32>, align 64
  %30 = alloca <16 x float>, align 64
  store <16 x i32> %0, <16 x i32>* %4, align 64, !tbaa !7
  store <16 x i32> %1, <16 x i32>* %5, align 64, !tbaa !7
  store <16 x i32>* %2, <16 x i32>** %6, align 64, !tbaa !7
  %31 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %32 = ashr <16 x i32> %31, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  store <16 x i32> %32, <16 x i32>* %27, align 64, !tbaa !7
  %33 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  %34 = ashr <16 x i32> %33, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  store <16 x i32> %34, <16 x i32>* %28, align 64, !tbaa !7
  %35 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %36 = load <16 x i32>, <16 x i32>* %27, align 64, !tbaa !7
  %37 = add <16 x i32> %35, %36
  %38 = load <16 x i32>, <16 x i32>* %27, align 64, !tbaa !7
  %39 = xor <16 x i32> %37, %38
  store <16 x i32> %39, <16 x i32>* %24, align 64, !tbaa !7
  %40 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  %41 = load <16 x i32>, <16 x i32>* %28, align 64, !tbaa !7
  %42 = add <16 x i32> %40, %41
  %43 = load <16 x i32>, <16 x i32>* %28, align 64, !tbaa !7
  %44 = xor <16 x i32> %42, %43
  store <16 x i32> %44, <16 x i32>* %25, align 64, !tbaa !7
  %45 = load <16 x i32>, <16 x i32>* %25, align 64, !tbaa !7
  %46 = uitofp <16 x i32> %45 to <16 x float>
  store <16 x float> %46, <16 x float>* %9, align 64, !tbaa !7
  %47 = load <16 x float>, <16 x float>* %9, align 64, !tbaa !7
  %48 = fptosi <16 x float> %47 to <16 x i32>
  store <16 x i32> %48, <16 x i32>* %17, align 64, !tbaa !7
  %49 = load <16 x i32>, <16 x i32>* %25, align 64, !tbaa !7
  %50 = load <16 x i32>, <16 x i32>* %17, align 64, !tbaa !7
  %51 = sub <16 x i32> %49, %50
  store <16 x i32> %51, <16 x i32>* %19, align 64, !tbaa !7
  %52 = load <16 x i32>, <16 x i32>* %19, align 64, !tbaa !7
  %53 = uitofp <16 x i32> %52 to <16 x float>
  store <16 x float> %53, <16 x float>* %10, align 64, !tbaa !7
  %54 = load <16 x i32>, <16 x i32>* %24, align 64, !tbaa !7
  %55 = uitofp <16 x i32> %54 to <16 x float>
  store <16 x float> %55, <16 x float>* %7, align 64, !tbaa !7
  %56 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %57 = fptosi <16 x float> %56 to <16 x i32>
  store <16 x i32> %57, <16 x i32>* %16, align 64, !tbaa !7
  %58 = load <16 x i32>, <16 x i32>* %24, align 64, !tbaa !7
  %59 = load <16 x i32>, <16 x i32>* %16, align 64, !tbaa !7
  %60 = sub <16 x i32> %58, %59
  store <16 x i32> %60, <16 x i32>* %18, align 64, !tbaa !7
  %61 = load <16 x i32>, <16 x i32>* %18, align 64, !tbaa !7
  %62 = uitofp <16 x i32> %61 to <16 x float>
  store <16 x float> %62, <16 x float>* %8, align 64, !tbaa !7
  %63 = load <16 x float>, <16 x float>* %9, align 64, !tbaa !7
  %64 = fdiv <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %63
  store <16 x float> %64, <16 x float>* %11, align 64, !tbaa !7
  %65 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %66 = insertelement <16 x float> undef, float %65, i32 0
  %67 = shufflevector <16 x float> %66, <16 x float> undef, <16 x i32> zeroinitializer
  store <16 x float> %67, <16 x float>* %30, align 64, !tbaa !7
  %68 = load <16 x float>, <16 x float>* %30, align 64, !tbaa !7
  %69 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %70 = fmul <16 x float> %68, %69
  %71 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %72 = fadd <16 x float> %71, %70
  store <16 x float> %72, <16 x float>* %11, align 64, !tbaa !7
  %73 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %74 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %75 = fmul <16 x float> %73, %74
  store <16 x float> %75, <16 x float>* %12, align 64, !tbaa !7
  %76 = load <16 x float>, <16 x float>* %12, align 64, !tbaa !7
  %77 = fptosi <16 x float> %76 to <16 x i32>
  store <16 x i32> %77, <16 x i32>* %20, align 64, !tbaa !7
  %78 = load <16 x i32>, <16 x i32>* %20, align 64, !tbaa !7
  %79 = uitofp <16 x i32> %78 to <16 x float>
  store <16 x float> %79, <16 x float>* %12, align 64, !tbaa !7
  %80 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %81 = load <16 x float>, <16 x float>* %9, align 64, !tbaa !7
  %82 = load <16 x float>, <16 x float>* %12, align 64, !tbaa !7
  %83 = fmul <16 x float> %81, %82
  %84 = fsub <16 x float> %80, %83
  store <16 x float> %84, <16 x float>* %14, align 64, !tbaa !7
  %85 = load <16 x float>, <16 x float>* %8, align 64, !tbaa !7
  %86 = load <16 x float>, <16 x float>* %10, align 64, !tbaa !7
  %87 = load <16 x float>, <16 x float>* %12, align 64, !tbaa !7
  %88 = fmul <16 x float> %86, %87
  %89 = fsub <16 x float> %85, %88
  store <16 x float> %89, <16 x float>* %15, align 64, !tbaa !7
  %90 = load <16 x float>, <16 x float>* %14, align 64, !tbaa !7
  %91 = load <16 x float>, <16 x float>* %15, align 64, !tbaa !7
  %92 = fadd <16 x float> %90, %91
  store <16 x float> %92, <16 x float>* %15, align 64, !tbaa !7
  %93 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %94 = load <16 x float>, <16 x float>* %15, align 64, !tbaa !7
  %95 = fmul <16 x float> %93, %94
  store <16 x float> %95, <16 x float>* %13, align 64, !tbaa !7
  %96 = load <16 x float>, <16 x float>* %13, align 64, !tbaa !7
  %97 = fptosi <16 x float> %96 to <16 x i32>
  store <16 x i32> %97, <16 x i32>* %21, align 64, !tbaa !7
  %98 = load <16 x i32>, <16 x i32>* %20, align 64, !tbaa !7
  %99 = load <16 x i32>, <16 x i32>* %21, align 64, !tbaa !7
  %100 = add <16 x i32> %98, %99
  store <16 x i32> %100, <16 x i32>* %22, align 64, !tbaa !7
  %101 = load <16 x i32>, <16 x i32>* %27, align 64, !tbaa !7
  %102 = load <16 x i32>, <16 x i32>* %28, align 64, !tbaa !7
  %103 = xor <16 x i32> %101, %102
  store <16 x i32> %103, <16 x i32>* %29, align 64, !tbaa !7
  %104 = load <16 x i32>, <16 x i32>* %24, align 64, !tbaa !7
  %105 = load <16 x i32>, <16 x i32>* %25, align 64, !tbaa !7
  %106 = load <16 x i32>, <16 x i32>* %22, align 64, !tbaa !7
  %107 = mul <16 x i32> %105, %106
  %108 = sub <16 x i32> %104, %107
  store <16 x i32> %108, <16 x i32>* %23, align 64, !tbaa !7
  %109 = load <16 x i32>, <16 x i32>* %23, align 64, !tbaa !7
  %110 = load <16 x i32>, <16 x i32>* %25, align 64, !tbaa !7
  %111 = icmp uge <16 x i32> %109, %110
  %112 = zext <16 x i1> %111 to <16 x i16>
  %113 = trunc <16 x i16> %112 to <16 x i1>
  %114 = select <16 x i1> %113, <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <16 x i32> zeroinitializer
  store <16 x i32> %114, <16 x i32>* %26, align 64, !tbaa !7
  %115 = load <16 x i32>, <16 x i32>* %26, align 64, !tbaa !7
  %116 = and <16 x i32> %115, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %117 = load <16 x i32>, <16 x i32>* %22, align 64, !tbaa !7
  %118 = add <16 x i32> %117, %116
  store <16 x i32> %118, <16 x i32>* %22, align 64, !tbaa !7
  %119 = load <16 x i32>, <16 x i32>* %25, align 64, !tbaa !7
  %120 = load <16 x i32>, <16 x i32>* %26, align 64, !tbaa !7
  %121 = and <16 x i32> %119, %120
  %122 = load <16 x i32>, <16 x i32>* %23, align 64, !tbaa !7
  %123 = sub <16 x i32> %122, %121
  store <16 x i32> %123, <16 x i32>* %23, align 64, !tbaa !7
  %124 = load <16 x i32>, <16 x i32>* %27, align 64, !tbaa !7
  %125 = load <16 x i32>, <16 x i32>* %23, align 64, !tbaa !7
  %126 = add <16 x i32> %124, %125
  %127 = load <16 x i32>, <16 x i32>* %27, align 64, !tbaa !7
  %128 = xor <16 x i32> %126, %127
  call void @llvm.genx.vstore.v16i32.p0v16i32(<16 x i32> %128, <16 x i32>* %2)
  %129 = load <16 x i32>, <16 x i32>* %29, align 64, !tbaa !7
  %130 = load <16 x i32>, <16 x i32>* %22, align 64, !tbaa !7
  %131 = add <16 x i32> %129, %130
  %132 = load <16 x i32>, <16 x i32>* %29, align 64, !tbaa !7
  %133 = xor <16 x i32> %131, %132
  store <16 x i32> %133, <16 x i32>* %22, align 64, !tbaa !7
  %134 = load <16 x i32>, <16 x i32>* %22, align 64, !tbaa !7
  ret <16 x i32> %134
}

; Function Attrs: noinline nounwind
define internal <32 x i32> @_Z24__cm_intrinsic_impl_sdivu2CMvb32_iS_(<32 x i32>, <32 x i32>) #19 {
  %3 = alloca <32 x i32>, align 128
  %4 = alloca <32 x i32>, align 128
  %5 = alloca <32 x i32>, align 128
  %6 = alloca <32 x i32>, align 128
  store <32 x i32> %0, <32 x i32>* %3, align 128, !tbaa !7
  store <32 x i32> %1, <32 x i32>* %4, align 128, !tbaa !7
  %7 = load <32 x i32>, <32 x i32>* %3, align 128, !tbaa !7
  %8 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %9 = call <32 x i32> @_ZN7details13__impl_divremILi32EEEu2CMvbT__iS1_S1_u2CMvrT__j(<32 x i32> %7, <32 x i32> %8, <32 x i32>* %5)
  store <32 x i32> %9, <32 x i32>* %6, align 128, !tbaa !7
  %10 = load <32 x i32>, <32 x i32>* %6, align 128, !tbaa !7
  ret <32 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <32 x i32> @_ZN7details13__impl_divremILi32EEEu2CMvbT__iS1_S1_u2CMvrT__j(<32 x i32>, <32 x i32>, <32 x i32>*) #20 comdat {
  %4 = alloca <32 x i32>, align 128
  %5 = alloca <32 x i32>, align 128
  %6 = alloca <32 x i32>*, align 128
  %7 = alloca <32 x float>, align 128
  %8 = alloca <32 x float>, align 128
  %9 = alloca <32 x float>, align 128
  %10 = alloca <32 x float>, align 128
  %11 = alloca <32 x float>, align 128
  %12 = alloca <32 x float>, align 128
  %13 = alloca <32 x float>, align 128
  %14 = alloca <32 x float>, align 128
  %15 = alloca <32 x float>, align 128
  %16 = alloca <32 x i32>, align 128
  %17 = alloca <32 x i32>, align 128
  %18 = alloca <32 x i32>, align 128
  %19 = alloca <32 x i32>, align 128
  %20 = alloca <32 x i32>, align 128
  %21 = alloca <32 x i32>, align 128
  %22 = alloca <32 x i32>, align 128
  %23 = alloca <32 x i32>, align 128
  %24 = alloca <32 x i32>, align 128
  %25 = alloca <32 x i32>, align 128
  %26 = alloca <32 x i32>, align 128
  %27 = alloca <32 x i32>, align 128
  %28 = alloca <32 x i32>, align 128
  %29 = alloca <32 x i32>, align 128
  %30 = alloca <32 x float>, align 128
  store <32 x i32> %0, <32 x i32>* %4, align 128, !tbaa !7
  store <32 x i32> %1, <32 x i32>* %5, align 128, !tbaa !7
  store <32 x i32>* %2, <32 x i32>** %6, align 128, !tbaa !7
  %31 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %32 = ashr <32 x i32> %31, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  store <32 x i32> %32, <32 x i32>* %27, align 128, !tbaa !7
  %33 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  %34 = ashr <32 x i32> %33, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  store <32 x i32> %34, <32 x i32>* %28, align 128, !tbaa !7
  %35 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %36 = load <32 x i32>, <32 x i32>* %27, align 128, !tbaa !7
  %37 = add <32 x i32> %35, %36
  %38 = load <32 x i32>, <32 x i32>* %27, align 128, !tbaa !7
  %39 = xor <32 x i32> %37, %38
  store <32 x i32> %39, <32 x i32>* %24, align 128, !tbaa !7
  %40 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  %41 = load <32 x i32>, <32 x i32>* %28, align 128, !tbaa !7
  %42 = add <32 x i32> %40, %41
  %43 = load <32 x i32>, <32 x i32>* %28, align 128, !tbaa !7
  %44 = xor <32 x i32> %42, %43
  store <32 x i32> %44, <32 x i32>* %25, align 128, !tbaa !7
  %45 = load <32 x i32>, <32 x i32>* %25, align 128, !tbaa !7
  %46 = uitofp <32 x i32> %45 to <32 x float>
  store <32 x float> %46, <32 x float>* %9, align 128, !tbaa !7
  %47 = load <32 x float>, <32 x float>* %9, align 128, !tbaa !7
  %48 = fptosi <32 x float> %47 to <32 x i32>
  store <32 x i32> %48, <32 x i32>* %17, align 128, !tbaa !7
  %49 = load <32 x i32>, <32 x i32>* %25, align 128, !tbaa !7
  %50 = load <32 x i32>, <32 x i32>* %17, align 128, !tbaa !7
  %51 = sub <32 x i32> %49, %50
  store <32 x i32> %51, <32 x i32>* %19, align 128, !tbaa !7
  %52 = load <32 x i32>, <32 x i32>* %19, align 128, !tbaa !7
  %53 = uitofp <32 x i32> %52 to <32 x float>
  store <32 x float> %53, <32 x float>* %10, align 128, !tbaa !7
  %54 = load <32 x i32>, <32 x i32>* %24, align 128, !tbaa !7
  %55 = uitofp <32 x i32> %54 to <32 x float>
  store <32 x float> %55, <32 x float>* %7, align 128, !tbaa !7
  %56 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %57 = fptosi <32 x float> %56 to <32 x i32>
  store <32 x i32> %57, <32 x i32>* %16, align 128, !tbaa !7
  %58 = load <32 x i32>, <32 x i32>* %24, align 128, !tbaa !7
  %59 = load <32 x i32>, <32 x i32>* %16, align 128, !tbaa !7
  %60 = sub <32 x i32> %58, %59
  store <32 x i32> %60, <32 x i32>* %18, align 128, !tbaa !7
  %61 = load <32 x i32>, <32 x i32>* %18, align 128, !tbaa !7
  %62 = uitofp <32 x i32> %61 to <32 x float>
  store <32 x float> %62, <32 x float>* %8, align 128, !tbaa !7
  %63 = load <32 x float>, <32 x float>* %9, align 128, !tbaa !7
  %64 = fdiv <32 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %63
  store <32 x float> %64, <32 x float>* %11, align 128, !tbaa !7
  %65 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %66 = insertelement <32 x float> undef, float %65, i32 0
  %67 = shufflevector <32 x float> %66, <32 x float> undef, <32 x i32> zeroinitializer
  store <32 x float> %67, <32 x float>* %30, align 128, !tbaa !7
  %68 = load <32 x float>, <32 x float>* %30, align 128, !tbaa !7
  %69 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %70 = fmul <32 x float> %68, %69
  %71 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %72 = fadd <32 x float> %71, %70
  store <32 x float> %72, <32 x float>* %11, align 128, !tbaa !7
  %73 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %74 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %75 = fmul <32 x float> %73, %74
  store <32 x float> %75, <32 x float>* %12, align 128, !tbaa !7
  %76 = load <32 x float>, <32 x float>* %12, align 128, !tbaa !7
  %77 = fptosi <32 x float> %76 to <32 x i32>
  store <32 x i32> %77, <32 x i32>* %20, align 128, !tbaa !7
  %78 = load <32 x i32>, <32 x i32>* %20, align 128, !tbaa !7
  %79 = uitofp <32 x i32> %78 to <32 x float>
  store <32 x float> %79, <32 x float>* %12, align 128, !tbaa !7
  %80 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %81 = load <32 x float>, <32 x float>* %9, align 128, !tbaa !7
  %82 = load <32 x float>, <32 x float>* %12, align 128, !tbaa !7
  %83 = fmul <32 x float> %81, %82
  %84 = fsub <32 x float> %80, %83
  store <32 x float> %84, <32 x float>* %14, align 128, !tbaa !7
  %85 = load <32 x float>, <32 x float>* %8, align 128, !tbaa !7
  %86 = load <32 x float>, <32 x float>* %10, align 128, !tbaa !7
  %87 = load <32 x float>, <32 x float>* %12, align 128, !tbaa !7
  %88 = fmul <32 x float> %86, %87
  %89 = fsub <32 x float> %85, %88
  store <32 x float> %89, <32 x float>* %15, align 128, !tbaa !7
  %90 = load <32 x float>, <32 x float>* %14, align 128, !tbaa !7
  %91 = load <32 x float>, <32 x float>* %15, align 128, !tbaa !7
  %92 = fadd <32 x float> %90, %91
  store <32 x float> %92, <32 x float>* %15, align 128, !tbaa !7
  %93 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %94 = load <32 x float>, <32 x float>* %15, align 128, !tbaa !7
  %95 = fmul <32 x float> %93, %94
  store <32 x float> %95, <32 x float>* %13, align 128, !tbaa !7
  %96 = load <32 x float>, <32 x float>* %13, align 128, !tbaa !7
  %97 = fptosi <32 x float> %96 to <32 x i32>
  store <32 x i32> %97, <32 x i32>* %21, align 128, !tbaa !7
  %98 = load <32 x i32>, <32 x i32>* %20, align 128, !tbaa !7
  %99 = load <32 x i32>, <32 x i32>* %21, align 128, !tbaa !7
  %100 = add <32 x i32> %98, %99
  store <32 x i32> %100, <32 x i32>* %22, align 128, !tbaa !7
  %101 = load <32 x i32>, <32 x i32>* %27, align 128, !tbaa !7
  %102 = load <32 x i32>, <32 x i32>* %28, align 128, !tbaa !7
  %103 = xor <32 x i32> %101, %102
  store <32 x i32> %103, <32 x i32>* %29, align 128, !tbaa !7
  %104 = load <32 x i32>, <32 x i32>* %24, align 128, !tbaa !7
  %105 = load <32 x i32>, <32 x i32>* %25, align 128, !tbaa !7
  %106 = load <32 x i32>, <32 x i32>* %22, align 128, !tbaa !7
  %107 = mul <32 x i32> %105, %106
  %108 = sub <32 x i32> %104, %107
  store <32 x i32> %108, <32 x i32>* %23, align 128, !tbaa !7
  %109 = load <32 x i32>, <32 x i32>* %23, align 128, !tbaa !7
  %110 = load <32 x i32>, <32 x i32>* %25, align 128, !tbaa !7
  %111 = icmp uge <32 x i32> %109, %110
  %112 = zext <32 x i1> %111 to <32 x i16>
  %113 = trunc <32 x i16> %112 to <32 x i1>
  %114 = select <32 x i1> %113, <32 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <32 x i32> zeroinitializer
  store <32 x i32> %114, <32 x i32>* %26, align 128, !tbaa !7
  %115 = load <32 x i32>, <32 x i32>* %26, align 128, !tbaa !7
  %116 = and <32 x i32> %115, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %117 = load <32 x i32>, <32 x i32>* %22, align 128, !tbaa !7
  %118 = add <32 x i32> %117, %116
  store <32 x i32> %118, <32 x i32>* %22, align 128, !tbaa !7
  %119 = load <32 x i32>, <32 x i32>* %25, align 128, !tbaa !7
  %120 = load <32 x i32>, <32 x i32>* %26, align 128, !tbaa !7
  %121 = and <32 x i32> %119, %120
  %122 = load <32 x i32>, <32 x i32>* %23, align 128, !tbaa !7
  %123 = sub <32 x i32> %122, %121
  store <32 x i32> %123, <32 x i32>* %23, align 128, !tbaa !7
  %124 = load <32 x i32>, <32 x i32>* %27, align 128, !tbaa !7
  %125 = load <32 x i32>, <32 x i32>* %23, align 128, !tbaa !7
  %126 = add <32 x i32> %124, %125
  %127 = load <32 x i32>, <32 x i32>* %27, align 128, !tbaa !7
  %128 = xor <32 x i32> %126, %127
  call void @llvm.genx.vstore.v32i32.p0v32i32(<32 x i32> %128, <32 x i32>* %2)
  %129 = load <32 x i32>, <32 x i32>* %29, align 128, !tbaa !7
  %130 = load <32 x i32>, <32 x i32>* %22, align 128, !tbaa !7
  %131 = add <32 x i32> %129, %130
  %132 = load <32 x i32>, <32 x i32>* %29, align 128, !tbaa !7
  %133 = xor <32 x i32> %131, %132
  store <32 x i32> %133, <32 x i32>* %22, align 128, !tbaa !7
  %134 = load <32 x i32>, <32 x i32>* %22, align 128, !tbaa !7
  ret <32 x i32> %134
}

; Function Attrs: noinline nounwind
define internal i32 @_Z24__cm_intrinsic_impl_sremii(i32, i32) #14 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca <1 x i32>, align 4
  %6 = alloca <1 x i32>, align 4
  %7 = alloca <1 x i32>, align 4
  store i32 %0, i32* %3, align 4, !tbaa !13
  store i32 %1, i32* %4, align 4, !tbaa !13
  %8 = load i32, i32* %3, align 4, !tbaa !13
  %9 = insertelement <1 x i32> undef, i32 %8, i32 0
  %10 = shufflevector <1 x i32> %9, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %10, <1 x i32>* %5, align 4, !tbaa !7
  %11 = load i32, i32* %4, align 4, !tbaa !13
  %12 = insertelement <1 x i32> undef, i32 %11, i32 0
  %13 = shufflevector <1 x i32> %12, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %13, <1 x i32>* %6, align 4, !tbaa !7
  %14 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %15 = load <1 x i32>, <1 x i32>* %6, align 4, !tbaa !7
  %16 = call <1 x i32> @_ZN7details13__impl_divremILi1EEEu2CMvbT__iS1_S1_u2CMvrT__j(<1 x i32> %14, <1 x i32> %15, <1 x i32>* %7)
  %17 = load <1 x i32>, <1 x i32>* %7, align 4, !tbaa !7
  %18 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v1i32.i16(<1 x i32> %17, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %19 = extractelement <1 x i32> %18, i32 0
  ret i32 %19
}

; Function Attrs: noinline nounwind
define internal <1 x i32> @_Z24__cm_intrinsic_impl_sremu2CMvb1_iS_(<1 x i32>, <1 x i32>) #14 {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x i32>, align 4
  %5 = alloca <1 x i32>, align 4
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %4, align 4, !tbaa !7
  %6 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %7 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %8 = call <1 x i32> @_ZN7details13__impl_divremILi1EEEu2CMvbT__iS1_S1_u2CMvrT__j(<1 x i32> %6, <1 x i32> %7, <1 x i32>* %5)
  %9 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  ret <1 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <2 x i32> @_Z24__cm_intrinsic_impl_sremu2CMvb2_iS_(<2 x i32>, <2 x i32>) #15 {
  %3 = alloca <2 x i32>, align 8
  %4 = alloca <2 x i32>, align 8
  %5 = alloca <2 x i32>, align 8
  store <2 x i32> %0, <2 x i32>* %3, align 8, !tbaa !7
  store <2 x i32> %1, <2 x i32>* %4, align 8, !tbaa !7
  %6 = load <2 x i32>, <2 x i32>* %3, align 8, !tbaa !7
  %7 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %8 = call <2 x i32> @_ZN7details13__impl_divremILi2EEEu2CMvbT__iS1_S1_u2CMvrT__j(<2 x i32> %6, <2 x i32> %7, <2 x i32>* %5)
  %9 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  ret <2 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <4 x i32> @_Z24__cm_intrinsic_impl_sremu2CMvb4_iS_(<4 x i32>, <4 x i32>) #16 {
  %3 = alloca <4 x i32>, align 16
  %4 = alloca <4 x i32>, align 16
  %5 = alloca <4 x i32>, align 16
  store <4 x i32> %0, <4 x i32>* %3, align 16, !tbaa !7
  store <4 x i32> %1, <4 x i32>* %4, align 16, !tbaa !7
  %6 = load <4 x i32>, <4 x i32>* %3, align 16, !tbaa !7
  %7 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %8 = call <4 x i32> @_ZN7details13__impl_divremILi4EEEu2CMvbT__iS1_S1_u2CMvrT__j(<4 x i32> %6, <4 x i32> %7, <4 x i32>* %5)
  %9 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  ret <4 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <8 x i32> @_Z24__cm_intrinsic_impl_sremu2CMvb8_iS_(<8 x i32>, <8 x i32>) #17 {
  %3 = alloca <8 x i32>, align 32
  %4 = alloca <8 x i32>, align 32
  %5 = alloca <8 x i32>, align 32
  store <8 x i32> %0, <8 x i32>* %3, align 32, !tbaa !7
  store <8 x i32> %1, <8 x i32>* %4, align 32, !tbaa !7
  %6 = load <8 x i32>, <8 x i32>* %3, align 32, !tbaa !7
  %7 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %8 = call <8 x i32> @_ZN7details13__impl_divremILi8EEEu2CMvbT__iS1_S1_u2CMvrT__j(<8 x i32> %6, <8 x i32> %7, <8 x i32>* %5)
  %9 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  ret <8 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <16 x i32> @_Z24__cm_intrinsic_impl_sremu2CMvb16_iS_(<16 x i32>, <16 x i32>) #18 {
  %3 = alloca <16 x i32>, align 64
  %4 = alloca <16 x i32>, align 64
  %5 = alloca <16 x i32>, align 64
  store <16 x i32> %0, <16 x i32>* %3, align 64, !tbaa !7
  store <16 x i32> %1, <16 x i32>* %4, align 64, !tbaa !7
  %6 = load <16 x i32>, <16 x i32>* %3, align 64, !tbaa !7
  %7 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %8 = call <16 x i32> @_ZN7details13__impl_divremILi16EEEu2CMvbT__iS1_S1_u2CMvrT__j(<16 x i32> %6, <16 x i32> %7, <16 x i32>* %5)
  %9 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  ret <16 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <32 x i32> @_Z24__cm_intrinsic_impl_sremu2CMvb32_iS_(<32 x i32>, <32 x i32>) #19 {
  %3 = alloca <32 x i32>, align 128
  %4 = alloca <32 x i32>, align 128
  %5 = alloca <32 x i32>, align 128
  store <32 x i32> %0, <32 x i32>* %3, align 128, !tbaa !7
  store <32 x i32> %1, <32 x i32>* %4, align 128, !tbaa !7
  %6 = load <32 x i32>, <32 x i32>* %3, align 128, !tbaa !7
  %7 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %8 = call <32 x i32> @_ZN7details13__impl_divremILi32EEEu2CMvbT__iS1_S1_u2CMvrT__j(<32 x i32> %6, <32 x i32> %7, <32 x i32>* %5)
  %9 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  ret <32 x i32> %9
}

; Function Attrs: noinline nounwind
define internal zeroext i8 @_Z24__cm_intrinsic_impl_udivhh(i8 zeroext, i8 zeroext) #21 {
  %3 = alloca i8, align 1
  %4 = alloca i8, align 1
  %5 = alloca <1 x i8>, align 1
  %6 = alloca <1 x i8>, align 1
  %7 = alloca <1 x i8>, align 1
  %8 = alloca <1 x i8>, align 1
  store i8 %0, i8* %3, align 1, !tbaa !7
  store i8 %1, i8* %4, align 1, !tbaa !7
  %9 = load i8, i8* %3, align 1, !tbaa !7
  %10 = insertelement <1 x i8> undef, i8 %9, i32 0
  %11 = shufflevector <1 x i8> %10, <1 x i8> undef, <1 x i32> zeroinitializer
  store <1 x i8> %11, <1 x i8>* %5, align 1, !tbaa !7
  %12 = load i8, i8* %4, align 1, !tbaa !7
  %13 = insertelement <1 x i8> undef, i8 %12, i32 0
  %14 = shufflevector <1 x i8> %13, <1 x i8> undef, <1 x i32> zeroinitializer
  store <1 x i8> %14, <1 x i8>* %6, align 1, !tbaa !7
  %15 = load <1 x i8>, <1 x i8>* %5, align 1, !tbaa !7
  %16 = load <1 x i8>, <1 x i8>* %6, align 1, !tbaa !7
  %17 = call <1 x i8> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__hS1_S1_u2CMvrT__h(<1 x i8> %15, <1 x i8> %16, <1 x i8>* %7)
  store <1 x i8> %17, <1 x i8>* %8, align 1, !tbaa !7
  %18 = load <1 x i8>, <1 x i8>* %8, align 1, !tbaa !7
  %19 = call <1 x i8> @llvm.genx.rdregioni.v1i8.v1i8.i16(<1 x i8> %18, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %20 = extractelement <1 x i8> %19, i32 0
  ret i8 %20
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i8> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__hS1_S1_u2CMvrT__h(<1 x i8>, <1 x i8>, <1 x i8>*) #1 comdat {
  %4 = alloca <1 x i8>, align 1
  %5 = alloca <1 x i8>, align 1
  %6 = alloca <1 x i8>*, align 1
  %7 = alloca <1 x i16>, align 2
  %8 = alloca <1 x i16>, align 2
  %9 = alloca <1 x i16>, align 2
  %10 = alloca <1 x i8>, align 1
  store <1 x i8> %0, <1 x i8>* %4, align 1, !tbaa !7
  store <1 x i8> %1, <1 x i8>* %5, align 1, !tbaa !7
  store <1 x i8>* %2, <1 x i8>** %6, align 1, !tbaa !7
  %11 = load <1 x i8>, <1 x i8>* %4, align 1, !tbaa !7
  %12 = zext <1 x i8> %11 to <1 x i16>
  store <1 x i16> %12, <1 x i16>* %7, align 2, !tbaa !7
  %13 = load <1 x i8>, <1 x i8>* %5, align 1, !tbaa !7
  %14 = zext <1 x i8> %13 to <1 x i16>
  store <1 x i16> %14, <1 x i16>* %8, align 2, !tbaa !7
  %15 = load <1 x i16>, <1 x i16>* %7, align 2, !tbaa !7
  %16 = load <1 x i16>, <1 x i16>* %8, align 2, !tbaa !7
  %17 = call <1 x i16> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__tS1_S1_u2CMvrT__t(<1 x i16> %15, <1 x i16> %16, <1 x i16>* %9)
  %18 = trunc <1 x i16> %17 to <1 x i8>
  store <1 x i8> %18, <1 x i8>* %10, align 1, !tbaa !7
  %19 = load <1 x i16>, <1 x i16>* %9, align 2, !tbaa !7
  %20 = trunc <1 x i16> %19 to <1 x i8>
  store <1 x i8> %20, <1 x i8>* %2
  %21 = load <1 x i8>, <1 x i8>* %10, align 1, !tbaa !7
  ret <1 x i8> %21
}

; Function Attrs: noinline nounwind
define internal <1 x i8> @_Z24__cm_intrinsic_impl_udivu2CMvb1_hS_(<1 x i8>, <1 x i8>) #21 {
  %3 = alloca <1 x i8>, align 1
  %4 = alloca <1 x i8>, align 1
  %5 = alloca <1 x i8>, align 1
  %6 = alloca <1 x i8>, align 1
  store <1 x i8> %0, <1 x i8>* %3, align 1, !tbaa !7
  store <1 x i8> %1, <1 x i8>* %4, align 1, !tbaa !7
  %7 = load <1 x i8>, <1 x i8>* %3, align 1, !tbaa !7
  %8 = load <1 x i8>, <1 x i8>* %4, align 1, !tbaa !7
  %9 = call <1 x i8> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__hS1_S1_u2CMvrT__h(<1 x i8> %7, <1 x i8> %8, <1 x i8>* %5)
  store <1 x i8> %9, <1 x i8>* %6, align 1, !tbaa !7
  %10 = load <1 x i8>, <1 x i8>* %6, align 1, !tbaa !7
  ret <1 x i8> %10
}

; Function Attrs: noinline nounwind
define internal <2 x i8> @_Z24__cm_intrinsic_impl_udivu2CMvb2_hS_(<2 x i8>, <2 x i8>) #22 {
  %3 = alloca <2 x i8>, align 2
  %4 = alloca <2 x i8>, align 2
  %5 = alloca <2 x i8>, align 2
  %6 = alloca <2 x i8>, align 2
  store <2 x i8> %0, <2 x i8>* %3, align 2, !tbaa !7
  store <2 x i8> %1, <2 x i8>* %4, align 2, !tbaa !7
  %7 = load <2 x i8>, <2 x i8>* %3, align 2, !tbaa !7
  %8 = load <2 x i8>, <2 x i8>* %4, align 2, !tbaa !7
  %9 = call <2 x i8> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__hS1_S1_u2CMvrT__h(<2 x i8> %7, <2 x i8> %8, <2 x i8>* %5)
  store <2 x i8> %9, <2 x i8>* %6, align 2, !tbaa !7
  %10 = load <2 x i8>, <2 x i8>* %6, align 2, !tbaa !7
  ret <2 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <2 x i8> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__hS1_S1_u2CMvrT__h(<2 x i8>, <2 x i8>, <2 x i8>*) #4 comdat {
  %4 = alloca <2 x i8>, align 2
  %5 = alloca <2 x i8>, align 2
  %6 = alloca <2 x i8>*, align 2
  %7 = alloca <2 x i16>, align 4
  %8 = alloca <2 x i16>, align 4
  %9 = alloca <2 x i16>, align 4
  %10 = alloca <2 x i8>, align 2
  store <2 x i8> %0, <2 x i8>* %4, align 2, !tbaa !7
  store <2 x i8> %1, <2 x i8>* %5, align 2, !tbaa !7
  store <2 x i8>* %2, <2 x i8>** %6, align 2, !tbaa !7
  %11 = load <2 x i8>, <2 x i8>* %4, align 2, !tbaa !7
  %12 = zext <2 x i8> %11 to <2 x i16>
  store <2 x i16> %12, <2 x i16>* %7, align 4, !tbaa !7
  %13 = load <2 x i8>, <2 x i8>* %5, align 2, !tbaa !7
  %14 = zext <2 x i8> %13 to <2 x i16>
  store <2 x i16> %14, <2 x i16>* %8, align 4, !tbaa !7
  %15 = load <2 x i16>, <2 x i16>* %7, align 4, !tbaa !7
  %16 = load <2 x i16>, <2 x i16>* %8, align 4, !tbaa !7
  %17 = call <2 x i16> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__tS1_S1_u2CMvrT__t(<2 x i16> %15, <2 x i16> %16, <2 x i16>* %9)
  %18 = trunc <2 x i16> %17 to <2 x i8>
  store <2 x i8> %18, <2 x i8>* %10, align 2, !tbaa !7
  %19 = load <2 x i16>, <2 x i16>* %9, align 4, !tbaa !7
  %20 = trunc <2 x i16> %19 to <2 x i8>
  call void @llvm.genx.vstore.v2i8.p0v2i8(<2 x i8> %20, <2 x i8>* %2)
  %21 = load <2 x i8>, <2 x i8>* %10, align 2, !tbaa !7
  ret <2 x i8> %21
}

; Function Attrs: noinline nounwind
define internal <4 x i8> @_Z24__cm_intrinsic_impl_udivu2CMvb4_hS_(<4 x i8>, <4 x i8>) #14 {
  %3 = alloca <4 x i8>, align 4
  %4 = alloca <4 x i8>, align 4
  %5 = alloca <4 x i8>, align 4
  %6 = alloca <4 x i8>, align 4
  store <4 x i8> %0, <4 x i8>* %3, align 4, !tbaa !7
  store <4 x i8> %1, <4 x i8>* %4, align 4, !tbaa !7
  %7 = load <4 x i8>, <4 x i8>* %3, align 4, !tbaa !7
  %8 = load <4 x i8>, <4 x i8>* %4, align 4, !tbaa !7
  %9 = call <4 x i8> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__hS1_S1_u2CMvrT__h(<4 x i8> %7, <4 x i8> %8, <4 x i8>* %5)
  store <4 x i8> %9, <4 x i8>* %6, align 4, !tbaa !7
  %10 = load <4 x i8>, <4 x i8>* %6, align 4, !tbaa !7
  ret <4 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <4 x i8> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__hS1_S1_u2CMvrT__h(<4 x i8>, <4 x i8>, <4 x i8>*) #6 comdat {
  %4 = alloca <4 x i8>, align 4
  %5 = alloca <4 x i8>, align 4
  %6 = alloca <4 x i8>*, align 4
  %7 = alloca <4 x i16>, align 8
  %8 = alloca <4 x i16>, align 8
  %9 = alloca <4 x i16>, align 8
  %10 = alloca <4 x i8>, align 4
  store <4 x i8> %0, <4 x i8>* %4, align 4, !tbaa !7
  store <4 x i8> %1, <4 x i8>* %5, align 4, !tbaa !7
  store <4 x i8>* %2, <4 x i8>** %6, align 4, !tbaa !7
  %11 = load <4 x i8>, <4 x i8>* %4, align 4, !tbaa !7
  %12 = zext <4 x i8> %11 to <4 x i16>
  store <4 x i16> %12, <4 x i16>* %7, align 8, !tbaa !7
  %13 = load <4 x i8>, <4 x i8>* %5, align 4, !tbaa !7
  %14 = zext <4 x i8> %13 to <4 x i16>
  store <4 x i16> %14, <4 x i16>* %8, align 8, !tbaa !7
  %15 = load <4 x i16>, <4 x i16>* %7, align 8, !tbaa !7
  %16 = load <4 x i16>, <4 x i16>* %8, align 8, !tbaa !7
  %17 = call <4 x i16> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__tS1_S1_u2CMvrT__t(<4 x i16> %15, <4 x i16> %16, <4 x i16>* %9)
  %18 = trunc <4 x i16> %17 to <4 x i8>
  store <4 x i8> %18, <4 x i8>* %10, align 4, !tbaa !7
  %19 = load <4 x i16>, <4 x i16>* %9, align 8, !tbaa !7
  %20 = trunc <4 x i16> %19 to <4 x i8>
  call void @llvm.genx.vstore.v4i8.p0v4i8(<4 x i8> %20, <4 x i8>* %2)
  %21 = load <4 x i8>, <4 x i8>* %10, align 4, !tbaa !7
  ret <4 x i8> %21
}

; Function Attrs: noinline nounwind
define dso_local <8 x i8> @_Z24__cm_intrinsic_impl_udivu2CMvb8_hS_(<8 x i8>, <8 x i8>) #15 {
  %3 = alloca <8 x i8>, align 8
  %4 = alloca <8 x i8>, align 8
  %5 = alloca <8 x i8>, align 8
  %6 = alloca <8 x i8>, align 8
  store <8 x i8> %0, <8 x i8>* %3, align 8, !tbaa !7
  store <8 x i8> %1, <8 x i8>* %4, align 8, !tbaa !7
  %7 = load <8 x i8>, <8 x i8>* %3, align 8, !tbaa !7
  %8 = load <8 x i8>, <8 x i8>* %4, align 8, !tbaa !7
  %9 = call <8 x i8> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__hS1_S1_u2CMvrT__h(<8 x i8> %7, <8 x i8> %8, <8 x i8>* %5)
  store <8 x i8> %9, <8 x i8>* %6, align 8, !tbaa !7
  %10 = load <8 x i8>, <8 x i8>* %6, align 8, !tbaa !7
  ret <8 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <8 x i8> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__hS1_S1_u2CMvrT__h(<8 x i8>, <8 x i8>, <8 x i8>*) #8 comdat {
  %4 = alloca <8 x i8>, align 8
  %5 = alloca <8 x i8>, align 8
  %6 = alloca <8 x i8>*, align 8
  %7 = alloca <8 x i16>, align 16
  %8 = alloca <8 x i16>, align 16
  %9 = alloca <8 x i16>, align 16
  %10 = alloca <8 x i8>, align 8
  store <8 x i8> %0, <8 x i8>* %4, align 8, !tbaa !7
  store <8 x i8> %1, <8 x i8>* %5, align 8, !tbaa !7
  store <8 x i8>* %2, <8 x i8>** %6, align 8, !tbaa !7
  %11 = load <8 x i8>, <8 x i8>* %4, align 8, !tbaa !7
  %12 = zext <8 x i8> %11 to <8 x i16>
  store <8 x i16> %12, <8 x i16>* %7, align 16, !tbaa !7
  %13 = load <8 x i8>, <8 x i8>* %5, align 8, !tbaa !7
  %14 = zext <8 x i8> %13 to <8 x i16>
  store <8 x i16> %14, <8 x i16>* %8, align 16, !tbaa !7
  %15 = load <8 x i16>, <8 x i16>* %7, align 16, !tbaa !7
  %16 = load <8 x i16>, <8 x i16>* %8, align 16, !tbaa !7
  %17 = call <8 x i16> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__tS1_S1_u2CMvrT__t(<8 x i16> %15, <8 x i16> %16, <8 x i16>* %9)
  %18 = trunc <8 x i16> %17 to <8 x i8>
  store <8 x i8> %18, <8 x i8>* %10, align 8, !tbaa !7
  %19 = load <8 x i16>, <8 x i16>* %9, align 16, !tbaa !7
  %20 = trunc <8 x i16> %19 to <8 x i8>
  call void @llvm.genx.vstore.v8i8.p0v8i8(<8 x i8> %20, <8 x i8>* %2)
  %21 = load <8 x i8>, <8 x i8>* %10, align 8, !tbaa !7
  ret <8 x i8> %21
}

; Function Attrs: noinline nounwind
define dso_local <16 x i8> @_Z24__cm_intrinsic_impl_udivu2CMvb16_hS_(<16 x i8>, <16 x i8>) #16 {
  %3 = alloca <16 x i8>, align 16
  %4 = alloca <16 x i8>, align 16
  %5 = alloca <16 x i8>, align 16
  %6 = alloca <16 x i8>, align 16
  store <16 x i8> %0, <16 x i8>* %3, align 16, !tbaa !7
  store <16 x i8> %1, <16 x i8>* %4, align 16, !tbaa !7
  %7 = load <16 x i8>, <16 x i8>* %3, align 16, !tbaa !7
  %8 = load <16 x i8>, <16 x i8>* %4, align 16, !tbaa !7
  %9 = call <16 x i8> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__hS1_S1_u2CMvrT__h(<16 x i8> %7, <16 x i8> %8, <16 x i8>* %5)
  store <16 x i8> %9, <16 x i8>* %6, align 16, !tbaa !7
  %10 = load <16 x i8>, <16 x i8>* %6, align 16, !tbaa !7
  ret <16 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <16 x i8> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__hS1_S1_u2CMvrT__h(<16 x i8>, <16 x i8>, <16 x i8>*) #10 comdat {
  %4 = alloca <16 x i8>, align 16
  %5 = alloca <16 x i8>, align 16
  %6 = alloca <16 x i8>*, align 16
  %7 = alloca <16 x i16>, align 32
  %8 = alloca <16 x i16>, align 32
  %9 = alloca <16 x i16>, align 32
  %10 = alloca <16 x i8>, align 16
  store <16 x i8> %0, <16 x i8>* %4, align 16, !tbaa !7
  store <16 x i8> %1, <16 x i8>* %5, align 16, !tbaa !7
  store <16 x i8>* %2, <16 x i8>** %6, align 16, !tbaa !7
  %11 = load <16 x i8>, <16 x i8>* %4, align 16, !tbaa !7
  %12 = zext <16 x i8> %11 to <16 x i16>
  store <16 x i16> %12, <16 x i16>* %7, align 32, !tbaa !7
  %13 = load <16 x i8>, <16 x i8>* %5, align 16, !tbaa !7
  %14 = zext <16 x i8> %13 to <16 x i16>
  store <16 x i16> %14, <16 x i16>* %8, align 32, !tbaa !7
  %15 = load <16 x i16>, <16 x i16>* %7, align 32, !tbaa !7
  %16 = load <16 x i16>, <16 x i16>* %8, align 32, !tbaa !7
  %17 = call <16 x i16> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__tS1_S1_u2CMvrT__t(<16 x i16> %15, <16 x i16> %16, <16 x i16>* %9)
  %18 = trunc <16 x i16> %17 to <16 x i8>
  store <16 x i8> %18, <16 x i8>* %10, align 16, !tbaa !7
  %19 = load <16 x i16>, <16 x i16>* %9, align 32, !tbaa !7
  %20 = trunc <16 x i16> %19 to <16 x i8>
  call void @llvm.genx.vstore.v16i8.p0v16i8(<16 x i8> %20, <16 x i8>* %2)
  %21 = load <16 x i8>, <16 x i8>* %10, align 16, !tbaa !7
  ret <16 x i8> %21
}

; Function Attrs: noinline nounwind
define internal <32 x i8> @_Z24__cm_intrinsic_impl_udivu2CMvb32_hS_(<32 x i8>, <32 x i8>) #17 {
  %3 = alloca <32 x i8>, align 32
  %4 = alloca <32 x i8>, align 32
  %5 = alloca <32 x i8>, align 32
  %6 = alloca <32 x i8>, align 32
  store <32 x i8> %0, <32 x i8>* %3, align 32, !tbaa !7
  store <32 x i8> %1, <32 x i8>* %4, align 32, !tbaa !7
  %7 = load <32 x i8>, <32 x i8>* %3, align 32, !tbaa !7
  %8 = load <32 x i8>, <32 x i8>* %4, align 32, !tbaa !7
  %9 = call <32 x i8> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__hS1_S1_u2CMvrT__h(<32 x i8> %7, <32 x i8> %8, <32 x i8>* %5)
  store <32 x i8> %9, <32 x i8>* %6, align 32, !tbaa !7
  %10 = load <32 x i8>, <32 x i8>* %6, align 32, !tbaa !7
  ret <32 x i8> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <32 x i8> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__hS1_S1_u2CMvrT__h(<32 x i8>, <32 x i8>, <32 x i8>*) #12 comdat {
  %4 = alloca <32 x i8>, align 32
  %5 = alloca <32 x i8>, align 32
  %6 = alloca <32 x i8>*, align 32
  %7 = alloca <32 x i16>, align 64
  %8 = alloca <32 x i16>, align 64
  %9 = alloca <32 x i16>, align 64
  %10 = alloca <32 x i8>, align 32
  store <32 x i8> %0, <32 x i8>* %4, align 32, !tbaa !7
  store <32 x i8> %1, <32 x i8>* %5, align 32, !tbaa !7
  store <32 x i8>* %2, <32 x i8>** %6, align 32, !tbaa !7
  %11 = load <32 x i8>, <32 x i8>* %4, align 32, !tbaa !7
  %12 = zext <32 x i8> %11 to <32 x i16>
  store <32 x i16> %12, <32 x i16>* %7, align 64, !tbaa !7
  %13 = load <32 x i8>, <32 x i8>* %5, align 32, !tbaa !7
  %14 = zext <32 x i8> %13 to <32 x i16>
  store <32 x i16> %14, <32 x i16>* %8, align 64, !tbaa !7
  %15 = load <32 x i16>, <32 x i16>* %7, align 64, !tbaa !7
  %16 = load <32 x i16>, <32 x i16>* %8, align 64, !tbaa !7
  %17 = call <32 x i16> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__tS1_S1_u2CMvrT__t(<32 x i16> %15, <32 x i16> %16, <32 x i16>* %9)
  %18 = trunc <32 x i16> %17 to <32 x i8>
  store <32 x i8> %18, <32 x i8>* %10, align 32, !tbaa !7
  %19 = load <32 x i16>, <32 x i16>* %9, align 64, !tbaa !7
  %20 = trunc <32 x i16> %19 to <32 x i8>
  call void @llvm.genx.vstore.v32i8.p0v32i8(<32 x i8> %20, <32 x i8>* %2)
  %21 = load <32 x i8>, <32 x i8>* %10, align 32, !tbaa !7
  ret <32 x i8> %21
}

; Function Attrs: noinline nounwind
define internal zeroext i8 @_Z24__cm_intrinsic_impl_uremhh(i8 zeroext, i8 zeroext) #21 {
  %3 = alloca i8, align 1
  %4 = alloca i8, align 1
  %5 = alloca <1 x i8>, align 1
  %6 = alloca <1 x i8>, align 1
  %7 = alloca <1 x i8>, align 1
  store i8 %0, i8* %3, align 1, !tbaa !7
  store i8 %1, i8* %4, align 1, !tbaa !7
  %8 = load i8, i8* %3, align 1, !tbaa !7
  %9 = insertelement <1 x i8> undef, i8 %8, i32 0
  %10 = shufflevector <1 x i8> %9, <1 x i8> undef, <1 x i32> zeroinitializer
  store <1 x i8> %10, <1 x i8>* %5, align 1, !tbaa !7
  %11 = load i8, i8* %4, align 1, !tbaa !7
  %12 = insertelement <1 x i8> undef, i8 %11, i32 0
  %13 = shufflevector <1 x i8> %12, <1 x i8> undef, <1 x i32> zeroinitializer
  store <1 x i8> %13, <1 x i8>* %6, align 1, !tbaa !7
  %14 = load <1 x i8>, <1 x i8>* %5, align 1, !tbaa !7
  %15 = load <1 x i8>, <1 x i8>* %6, align 1, !tbaa !7
  %16 = call <1 x i8> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__hS1_S1_u2CMvrT__h(<1 x i8> %14, <1 x i8> %15, <1 x i8>* %7)
  %17 = load <1 x i8>, <1 x i8>* %7, align 1, !tbaa !7
  %18 = call <1 x i8> @llvm.genx.rdregioni.v1i8.v1i8.i16(<1 x i8> %17, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %19 = extractelement <1 x i8> %18, i32 0
  ret i8 %19
}

; Function Attrs: noinline nounwind
define internal <1 x i8> @_Z24__cm_intrinsic_impl_uremu2CMvb1_hS_(<1 x i8>, <1 x i8>) #21 {
  %3 = alloca <1 x i8>, align 1
  %4 = alloca <1 x i8>, align 1
  %5 = alloca <1 x i8>, align 1
  store <1 x i8> %0, <1 x i8>* %3, align 1, !tbaa !7
  store <1 x i8> %1, <1 x i8>* %4, align 1, !tbaa !7
  %6 = load <1 x i8>, <1 x i8>* %3, align 1, !tbaa !7
  %7 = load <1 x i8>, <1 x i8>* %4, align 1, !tbaa !7
  %8 = call <1 x i8> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__hS1_S1_u2CMvrT__h(<1 x i8> %6, <1 x i8> %7, <1 x i8>* %5)
  %9 = load <1 x i8>, <1 x i8>* %5, align 1, !tbaa !7
  ret <1 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <2 x i8> @_Z24__cm_intrinsic_impl_uremu2CMvb2_hS_(<2 x i8>, <2 x i8>) #22 {
  %3 = alloca <2 x i8>, align 2
  %4 = alloca <2 x i8>, align 2
  %5 = alloca <2 x i8>, align 2
  store <2 x i8> %0, <2 x i8>* %3, align 2, !tbaa !7
  store <2 x i8> %1, <2 x i8>* %4, align 2, !tbaa !7
  %6 = load <2 x i8>, <2 x i8>* %3, align 2, !tbaa !7
  %7 = load <2 x i8>, <2 x i8>* %4, align 2, !tbaa !7
  %8 = call <2 x i8> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__hS1_S1_u2CMvrT__h(<2 x i8> %6, <2 x i8> %7, <2 x i8>* %5)
  %9 = load <2 x i8>, <2 x i8>* %5, align 2, !tbaa !7
  ret <2 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <4 x i8> @_Z24__cm_intrinsic_impl_uremu2CMvb4_hS_(<4 x i8>, <4 x i8>) #14 {
  %3 = alloca <4 x i8>, align 4
  %4 = alloca <4 x i8>, align 4
  %5 = alloca <4 x i8>, align 4
  store <4 x i8> %0, <4 x i8>* %3, align 4, !tbaa !7
  store <4 x i8> %1, <4 x i8>* %4, align 4, !tbaa !7
  %6 = load <4 x i8>, <4 x i8>* %3, align 4, !tbaa !7
  %7 = load <4 x i8>, <4 x i8>* %4, align 4, !tbaa !7
  %8 = call <4 x i8> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__hS1_S1_u2CMvrT__h(<4 x i8> %6, <4 x i8> %7, <4 x i8>* %5)
  %9 = load <4 x i8>, <4 x i8>* %5, align 4, !tbaa !7
  ret <4 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <8 x i8> @_Z24__cm_intrinsic_impl_uremu2CMvb8_hS_(<8 x i8>, <8 x i8>) #15 {
  %3 = alloca <8 x i8>, align 8
  %4 = alloca <8 x i8>, align 8
  %5 = alloca <8 x i8>, align 8
  store <8 x i8> %0, <8 x i8>* %3, align 8, !tbaa !7
  store <8 x i8> %1, <8 x i8>* %4, align 8, !tbaa !7
  %6 = load <8 x i8>, <8 x i8>* %3, align 8, !tbaa !7
  %7 = load <8 x i8>, <8 x i8>* %4, align 8, !tbaa !7
  %8 = call <8 x i8> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__hS1_S1_u2CMvrT__h(<8 x i8> %6, <8 x i8> %7, <8 x i8>* %5)
  %9 = load <8 x i8>, <8 x i8>* %5, align 8, !tbaa !7
  ret <8 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <16 x i8> @_Z24__cm_intrinsic_impl_uremu2CMvb16_hS_(<16 x i8>, <16 x i8>) #16 {
  %3 = alloca <16 x i8>, align 16
  %4 = alloca <16 x i8>, align 16
  %5 = alloca <16 x i8>, align 16
  store <16 x i8> %0, <16 x i8>* %3, align 16, !tbaa !7
  store <16 x i8> %1, <16 x i8>* %4, align 16, !tbaa !7
  %6 = load <16 x i8>, <16 x i8>* %3, align 16, !tbaa !7
  %7 = load <16 x i8>, <16 x i8>* %4, align 16, !tbaa !7
  %8 = call <16 x i8> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__hS1_S1_u2CMvrT__h(<16 x i8> %6, <16 x i8> %7, <16 x i8>* %5)
  %9 = load <16 x i8>, <16 x i8>* %5, align 16, !tbaa !7
  ret <16 x i8> %9
}

; Function Attrs: noinline nounwind
define internal <32 x i8> @_Z24__cm_intrinsic_impl_uremu2CMvb32_hS_(<32 x i8>, <32 x i8>) #17 {
  %3 = alloca <32 x i8>, align 32
  %4 = alloca <32 x i8>, align 32
  %5 = alloca <32 x i8>, align 32
  store <32 x i8> %0, <32 x i8>* %3, align 32, !tbaa !7
  store <32 x i8> %1, <32 x i8>* %4, align 32, !tbaa !7
  %6 = load <32 x i8>, <32 x i8>* %3, align 32, !tbaa !7
  %7 = load <32 x i8>, <32 x i8>* %4, align 32, !tbaa !7
  %8 = call <32 x i8> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__hS1_S1_u2CMvrT__h(<32 x i8> %6, <32 x i8> %7, <32 x i8>* %5)
  %9 = load <32 x i8>, <32 x i8>* %5, align 32, !tbaa !7
  ret <32 x i8> %9
}

; Function Attrs: noinline nounwind
define internal zeroext i16 @_Z24__cm_intrinsic_impl_udivtt(i16 zeroext, i16 zeroext) #22 {
  %3 = alloca i16, align 2
  %4 = alloca i16, align 2
  %5 = alloca <1 x i16>, align 2
  %6 = alloca <1 x i16>, align 2
  %7 = alloca <1 x i16>, align 2
  %8 = alloca <1 x i16>, align 2
  store i16 %0, i16* %3, align 2, !tbaa !11
  store i16 %1, i16* %4, align 2, !tbaa !11
  %9 = load i16, i16* %3, align 2, !tbaa !11
  %10 = insertelement <1 x i16> undef, i16 %9, i32 0
  %11 = shufflevector <1 x i16> %10, <1 x i16> undef, <1 x i32> zeroinitializer
  store <1 x i16> %11, <1 x i16>* %5, align 2, !tbaa !7
  %12 = load i16, i16* %4, align 2, !tbaa !11
  %13 = insertelement <1 x i16> undef, i16 %12, i32 0
  %14 = shufflevector <1 x i16> %13, <1 x i16> undef, <1 x i32> zeroinitializer
  store <1 x i16> %14, <1 x i16>* %6, align 2, !tbaa !7
  %15 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  %16 = load <1 x i16>, <1 x i16>* %6, align 2, !tbaa !7
  %17 = call <1 x i16> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__tS1_S1_u2CMvrT__t(<1 x i16> %15, <1 x i16> %16, <1 x i16>* %7)
  store <1 x i16> %17, <1 x i16>* %8, align 2, !tbaa !7
  %18 = load <1 x i16>, <1 x i16>* %8, align 2, !tbaa !7
  %19 = call <1 x i16> @llvm.genx.rdregioni.v1i16.v1i16.i16(<1 x i16> %18, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %20 = extractelement <1 x i16> %19, i32 0
  ret i16 %20
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i16> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__tS1_S1_u2CMvrT__t(<1 x i16>, <1 x i16>, <1 x i16>*) #1 comdat {
  %4 = alloca <1 x i16>, align 2
  %5 = alloca <1 x i16>, align 2
  %6 = alloca <1 x i16>*, align 2
  %7 = alloca <1 x float>, align 4
  %8 = alloca <1 x float>, align 4
  %9 = alloca <1 x float>, align 4
  %10 = alloca <1 x float>, align 4
  %11 = alloca <1 x i16>, align 2
  store <1 x i16> %0, <1 x i16>* %4, align 2, !tbaa !7
  store <1 x i16> %1, <1 x i16>* %5, align 2, !tbaa !7
  store <1 x i16>* %2, <1 x i16>** %6, align 2, !tbaa !7
  %12 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  %13 = uitofp <1 x i16> %12 to <1 x float>
  store <1 x float> %13, <1 x float>* %8, align 4, !tbaa !7
  %14 = load <1 x i16>, <1 x i16>* %4, align 2, !tbaa !7
  %15 = uitofp <1 x i16> %14 to <1 x float>
  store <1 x float> %15, <1 x float>* %7, align 4, !tbaa !7
  %16 = load <1 x float>, <1 x float>* %8, align 4, !tbaa !7
  %17 = fdiv <1 x float> <float 1.000000e+00>, %16
  store <1 x float> %17, <1 x float>* %9, align 4, !tbaa !7
  %18 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <1 x float> undef, float %19, i32 0
  %21 = shufflevector <1 x float> %20, <1 x float> undef, <1 x i32> zeroinitializer
  %22 = fmul <1 x float> %18, %21
  store <1 x float> %22, <1 x float>* %7, align 4, !tbaa !7
  %23 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %24 = load <1 x float>, <1 x float>* %9, align 4, !tbaa !7
  %25 = fmul <1 x float> %23, %24
  store <1 x float> %25, <1 x float>* %10, align 4, !tbaa !7
  %26 = load <1 x float>, <1 x float>* %10, align 4, !tbaa !7
  %27 = fptosi <1 x float> %26 to <1 x i32>
  %28 = trunc <1 x i32> %27 to <1 x i16>
  store <1 x i16> %28, <1 x i16>* %11, align 2, !tbaa !7
  %29 = load <1 x i16>, <1 x i16>* %4, align 2, !tbaa !7
  %30 = zext <1 x i16> %29 to <1 x i32>
  %31 = load <1 x i16>, <1 x i16>* %11, align 2, !tbaa !7
  %32 = zext <1 x i16> %31 to <1 x i32>
  %33 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  %34 = zext <1 x i16> %33 to <1 x i32>
  %35 = mul <1 x i32> %32, %34
  %36 = sub <1 x i32> %30, %35
  %37 = trunc <1 x i32> %36 to <1 x i16>
  store <1 x i16> %37, <1 x i16>* %2
  %38 = load <1 x i16>, <1 x i16>* %11, align 2, !tbaa !7
  ret <1 x i16> %38
}

; Function Attrs: noinline nounwind
define internal <1 x i16> @_Z24__cm_intrinsic_impl_udivu2CMvb1_tS_(<1 x i16>, <1 x i16>) #22 {
  %3 = alloca <1 x i16>, align 2
  %4 = alloca <1 x i16>, align 2
  %5 = alloca <1 x i16>, align 2
  %6 = alloca <1 x i16>, align 2
  store <1 x i16> %0, <1 x i16>* %3, align 2, !tbaa !7
  store <1 x i16> %1, <1 x i16>* %4, align 2, !tbaa !7
  %7 = load <1 x i16>, <1 x i16>* %3, align 2, !tbaa !7
  %8 = load <1 x i16>, <1 x i16>* %4, align 2, !tbaa !7
  %9 = call <1 x i16> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__tS1_S1_u2CMvrT__t(<1 x i16> %7, <1 x i16> %8, <1 x i16>* %5)
  store <1 x i16> %9, <1 x i16>* %6, align 2, !tbaa !7
  %10 = load <1 x i16>, <1 x i16>* %6, align 2, !tbaa !7
  ret <1 x i16> %10
}

; Function Attrs: noinline nounwind
define internal <2 x i16> @_Z24__cm_intrinsic_impl_udivu2CMvb2_tS_(<2 x i16>, <2 x i16>) #14 {
  %3 = alloca <2 x i16>, align 4
  %4 = alloca <2 x i16>, align 4
  %5 = alloca <2 x i16>, align 4
  %6 = alloca <2 x i16>, align 4
  store <2 x i16> %0, <2 x i16>* %3, align 4, !tbaa !7
  store <2 x i16> %1, <2 x i16>* %4, align 4, !tbaa !7
  %7 = load <2 x i16>, <2 x i16>* %3, align 4, !tbaa !7
  %8 = load <2 x i16>, <2 x i16>* %4, align 4, !tbaa !7
  %9 = call <2 x i16> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__tS1_S1_u2CMvrT__t(<2 x i16> %7, <2 x i16> %8, <2 x i16>* %5)
  store <2 x i16> %9, <2 x i16>* %6, align 4, !tbaa !7
  %10 = load <2 x i16>, <2 x i16>* %6, align 4, !tbaa !7
  ret <2 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <2 x i16> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__tS1_S1_u2CMvrT__t(<2 x i16>, <2 x i16>, <2 x i16>*) #4 comdat {
  %4 = alloca <2 x i16>, align 4
  %5 = alloca <2 x i16>, align 4
  %6 = alloca <2 x i16>*, align 4
  %7 = alloca <2 x float>, align 8
  %8 = alloca <2 x float>, align 8
  %9 = alloca <2 x float>, align 8
  %10 = alloca <2 x float>, align 8
  %11 = alloca <2 x i16>, align 4
  store <2 x i16> %0, <2 x i16>* %4, align 4, !tbaa !7
  store <2 x i16> %1, <2 x i16>* %5, align 4, !tbaa !7
  store <2 x i16>* %2, <2 x i16>** %6, align 4, !tbaa !7
  %12 = load <2 x i16>, <2 x i16>* %5, align 4, !tbaa !7
  %13 = uitofp <2 x i16> %12 to <2 x float>
  store <2 x float> %13, <2 x float>* %8, align 8, !tbaa !7
  %14 = load <2 x i16>, <2 x i16>* %4, align 4, !tbaa !7
  %15 = uitofp <2 x i16> %14 to <2 x float>
  store <2 x float> %15, <2 x float>* %7, align 8, !tbaa !7
  %16 = load <2 x float>, <2 x float>* %8, align 8, !tbaa !7
  %17 = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %16
  store <2 x float> %17, <2 x float>* %9, align 8, !tbaa !7
  %18 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <2 x float> undef, float %19, i32 0
  %21 = shufflevector <2 x float> %20, <2 x float> undef, <2 x i32> zeroinitializer
  %22 = fmul <2 x float> %18, %21
  store <2 x float> %22, <2 x float>* %7, align 8, !tbaa !7
  %23 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %24 = load <2 x float>, <2 x float>* %9, align 8, !tbaa !7
  %25 = fmul <2 x float> %23, %24
  store <2 x float> %25, <2 x float>* %10, align 8, !tbaa !7
  %26 = load <2 x float>, <2 x float>* %10, align 8, !tbaa !7
  %27 = fptosi <2 x float> %26 to <2 x i32>
  %28 = trunc <2 x i32> %27 to <2 x i16>
  store <2 x i16> %28, <2 x i16>* %11, align 4, !tbaa !7
  %29 = load <2 x i16>, <2 x i16>* %4, align 4, !tbaa !7
  %30 = zext <2 x i16> %29 to <2 x i32>
  %31 = load <2 x i16>, <2 x i16>* %11, align 4, !tbaa !7
  %32 = zext <2 x i16> %31 to <2 x i32>
  %33 = load <2 x i16>, <2 x i16>* %5, align 4, !tbaa !7
  %34 = zext <2 x i16> %33 to <2 x i32>
  %35 = mul <2 x i32> %32, %34
  %36 = sub <2 x i32> %30, %35
  %37 = trunc <2 x i32> %36 to <2 x i16>
  call void @llvm.genx.vstore.v2i16.p0v2i16(<2 x i16> %37, <2 x i16>* %2)
  %38 = load <2 x i16>, <2 x i16>* %11, align 4, !tbaa !7
  ret <2 x i16> %38
}

; Function Attrs: noinline nounwind
define internal <4 x i16> @_Z24__cm_intrinsic_impl_udivu2CMvb4_tS_(<4 x i16>, <4 x i16>) #15 {
  %3 = alloca <4 x i16>, align 8
  %4 = alloca <4 x i16>, align 8
  %5 = alloca <4 x i16>, align 8
  %6 = alloca <4 x i16>, align 8
  store <4 x i16> %0, <4 x i16>* %3, align 8, !tbaa !7
  store <4 x i16> %1, <4 x i16>* %4, align 8, !tbaa !7
  %7 = load <4 x i16>, <4 x i16>* %3, align 8, !tbaa !7
  %8 = load <4 x i16>, <4 x i16>* %4, align 8, !tbaa !7
  %9 = call <4 x i16> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__tS1_S1_u2CMvrT__t(<4 x i16> %7, <4 x i16> %8, <4 x i16>* %5)
  store <4 x i16> %9, <4 x i16>* %6, align 8, !tbaa !7
  %10 = load <4 x i16>, <4 x i16>* %6, align 8, !tbaa !7
  ret <4 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <4 x i16> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__tS1_S1_u2CMvrT__t(<4 x i16>, <4 x i16>, <4 x i16>*) #6 comdat {
  %4 = alloca <4 x i16>, align 8
  %5 = alloca <4 x i16>, align 8
  %6 = alloca <4 x i16>*, align 8
  %7 = alloca <4 x float>, align 16
  %8 = alloca <4 x float>, align 16
  %9 = alloca <4 x float>, align 16
  %10 = alloca <4 x float>, align 16
  %11 = alloca <4 x i16>, align 8
  store <4 x i16> %0, <4 x i16>* %4, align 8, !tbaa !7
  store <4 x i16> %1, <4 x i16>* %5, align 8, !tbaa !7
  store <4 x i16>* %2, <4 x i16>** %6, align 8, !tbaa !7
  %12 = load <4 x i16>, <4 x i16>* %5, align 8, !tbaa !7
  %13 = uitofp <4 x i16> %12 to <4 x float>
  store <4 x float> %13, <4 x float>* %8, align 16, !tbaa !7
  %14 = load <4 x i16>, <4 x i16>* %4, align 8, !tbaa !7
  %15 = uitofp <4 x i16> %14 to <4 x float>
  store <4 x float> %15, <4 x float>* %7, align 16, !tbaa !7
  %16 = load <4 x float>, <4 x float>* %8, align 16, !tbaa !7
  %17 = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %16
  store <4 x float> %17, <4 x float>* %9, align 16, !tbaa !7
  %18 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <4 x float> undef, float %19, i32 0
  %21 = shufflevector <4 x float> %20, <4 x float> undef, <4 x i32> zeroinitializer
  %22 = fmul <4 x float> %18, %21
  store <4 x float> %22, <4 x float>* %7, align 16, !tbaa !7
  %23 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %24 = load <4 x float>, <4 x float>* %9, align 16, !tbaa !7
  %25 = fmul <4 x float> %23, %24
  store <4 x float> %25, <4 x float>* %10, align 16, !tbaa !7
  %26 = load <4 x float>, <4 x float>* %10, align 16, !tbaa !7
  %27 = fptosi <4 x float> %26 to <4 x i32>
  %28 = trunc <4 x i32> %27 to <4 x i16>
  store <4 x i16> %28, <4 x i16>* %11, align 8, !tbaa !7
  %29 = load <4 x i16>, <4 x i16>* %4, align 8, !tbaa !7
  %30 = zext <4 x i16> %29 to <4 x i32>
  %31 = load <4 x i16>, <4 x i16>* %11, align 8, !tbaa !7
  %32 = zext <4 x i16> %31 to <4 x i32>
  %33 = load <4 x i16>, <4 x i16>* %5, align 8, !tbaa !7
  %34 = zext <4 x i16> %33 to <4 x i32>
  %35 = mul <4 x i32> %32, %34
  %36 = sub <4 x i32> %30, %35
  %37 = trunc <4 x i32> %36 to <4 x i16>
  call void @llvm.genx.vstore.v4i16.p0v4i16(<4 x i16> %37, <4 x i16>* %2)
  %38 = load <4 x i16>, <4 x i16>* %11, align 8, !tbaa !7
  ret <4 x i16> %38
}

; Function Attrs: noinline nounwind
define dso_local <8 x i16> @_Z24__cm_intrinsic_impl_udivu2CMvb8_tS_(<8 x i16>, <8 x i16>) #16 {
  %3 = alloca <8 x i16>, align 16
  %4 = alloca <8 x i16>, align 16
  %5 = alloca <8 x i16>, align 16
  %6 = alloca <8 x i16>, align 16
  store <8 x i16> %0, <8 x i16>* %3, align 16, !tbaa !7
  store <8 x i16> %1, <8 x i16>* %4, align 16, !tbaa !7
  %7 = load <8 x i16>, <8 x i16>* %3, align 16, !tbaa !7
  %8 = load <8 x i16>, <8 x i16>* %4, align 16, !tbaa !7
  %9 = call <8 x i16> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__tS1_S1_u2CMvrT__t(<8 x i16> %7, <8 x i16> %8, <8 x i16>* %5)
  store <8 x i16> %9, <8 x i16>* %6, align 16, !tbaa !7
  %10 = load <8 x i16>, <8 x i16>* %6, align 16, !tbaa !7
  ret <8 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <8 x i16> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__tS1_S1_u2CMvrT__t(<8 x i16>, <8 x i16>, <8 x i16>*) #8 comdat {
  %4 = alloca <8 x i16>, align 16
  %5 = alloca <8 x i16>, align 16
  %6 = alloca <8 x i16>*, align 16
  %7 = alloca <8 x float>, align 32
  %8 = alloca <8 x float>, align 32
  %9 = alloca <8 x float>, align 32
  %10 = alloca <8 x float>, align 32
  %11 = alloca <8 x i16>, align 16
  store <8 x i16> %0, <8 x i16>* %4, align 16, !tbaa !7
  store <8 x i16> %1, <8 x i16>* %5, align 16, !tbaa !7
  store <8 x i16>* %2, <8 x i16>** %6, align 16, !tbaa !7
  %12 = load <8 x i16>, <8 x i16>* %5, align 16, !tbaa !7
  %13 = uitofp <8 x i16> %12 to <8 x float>
  store <8 x float> %13, <8 x float>* %8, align 32, !tbaa !7
  %14 = load <8 x i16>, <8 x i16>* %4, align 16, !tbaa !7
  %15 = uitofp <8 x i16> %14 to <8 x float>
  store <8 x float> %15, <8 x float>* %7, align 32, !tbaa !7
  %16 = load <8 x float>, <8 x float>* %8, align 32, !tbaa !7
  %17 = fdiv <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %16
  store <8 x float> %17, <8 x float>* %9, align 32, !tbaa !7
  %18 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <8 x float> undef, float %19, i32 0
  %21 = shufflevector <8 x float> %20, <8 x float> undef, <8 x i32> zeroinitializer
  %22 = fmul <8 x float> %18, %21
  store <8 x float> %22, <8 x float>* %7, align 32, !tbaa !7
  %23 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %24 = load <8 x float>, <8 x float>* %9, align 32, !tbaa !7
  %25 = fmul <8 x float> %23, %24
  store <8 x float> %25, <8 x float>* %10, align 32, !tbaa !7
  %26 = load <8 x float>, <8 x float>* %10, align 32, !tbaa !7
  %27 = fptosi <8 x float> %26 to <8 x i32>
  %28 = trunc <8 x i32> %27 to <8 x i16>
  store <8 x i16> %28, <8 x i16>* %11, align 16, !tbaa !7
  %29 = load <8 x i16>, <8 x i16>* %4, align 16, !tbaa !7
  %30 = zext <8 x i16> %29 to <8 x i32>
  %31 = load <8 x i16>, <8 x i16>* %11, align 16, !tbaa !7
  %32 = zext <8 x i16> %31 to <8 x i32>
  %33 = load <8 x i16>, <8 x i16>* %5, align 16, !tbaa !7
  %34 = zext <8 x i16> %33 to <8 x i32>
  %35 = mul <8 x i32> %32, %34
  %36 = sub <8 x i32> %30, %35
  %37 = trunc <8 x i32> %36 to <8 x i16>
  call void @llvm.genx.vstore.v8i16.p0v8i16(<8 x i16> %37, <8 x i16>* %2)
  %38 = load <8 x i16>, <8 x i16>* %11, align 16, !tbaa !7
  ret <8 x i16> %38
}

; Function Attrs: noinline nounwind
define dso_local <16 x i16> @_Z24__cm_intrinsic_impl_udivu2CMvb16_tS_(<16 x i16>, <16 x i16>) #17 {
  %3 = alloca <16 x i16>, align 32
  %4 = alloca <16 x i16>, align 32
  %5 = alloca <16 x i16>, align 32
  %6 = alloca <16 x i16>, align 32
  store <16 x i16> %0, <16 x i16>* %3, align 32, !tbaa !7
  store <16 x i16> %1, <16 x i16>* %4, align 32, !tbaa !7
  %7 = load <16 x i16>, <16 x i16>* %3, align 32, !tbaa !7
  %8 = load <16 x i16>, <16 x i16>* %4, align 32, !tbaa !7
  %9 = call <16 x i16> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__tS1_S1_u2CMvrT__t(<16 x i16> %7, <16 x i16> %8, <16 x i16>* %5)
  store <16 x i16> %9, <16 x i16>* %6, align 32, !tbaa !7
  %10 = load <16 x i16>, <16 x i16>* %6, align 32, !tbaa !7
  ret <16 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <16 x i16> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__tS1_S1_u2CMvrT__t(<16 x i16>, <16 x i16>, <16 x i16>*) #10 comdat {
  %4 = alloca <16 x i16>, align 32
  %5 = alloca <16 x i16>, align 32
  %6 = alloca <16 x i16>*, align 32
  %7 = alloca <16 x float>, align 64
  %8 = alloca <16 x float>, align 64
  %9 = alloca <16 x float>, align 64
  %10 = alloca <16 x float>, align 64
  %11 = alloca <16 x i16>, align 32
  store <16 x i16> %0, <16 x i16>* %4, align 32, !tbaa !7
  store <16 x i16> %1, <16 x i16>* %5, align 32, !tbaa !7
  store <16 x i16>* %2, <16 x i16>** %6, align 32, !tbaa !7
  %12 = load <16 x i16>, <16 x i16>* %5, align 32, !tbaa !7
  %13 = uitofp <16 x i16> %12 to <16 x float>
  store <16 x float> %13, <16 x float>* %8, align 64, !tbaa !7
  %14 = load <16 x i16>, <16 x i16>* %4, align 32, !tbaa !7
  %15 = uitofp <16 x i16> %14 to <16 x float>
  store <16 x float> %15, <16 x float>* %7, align 64, !tbaa !7
  %16 = load <16 x float>, <16 x float>* %8, align 64, !tbaa !7
  %17 = fdiv <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %16
  store <16 x float> %17, <16 x float>* %9, align 64, !tbaa !7
  %18 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <16 x float> undef, float %19, i32 0
  %21 = shufflevector <16 x float> %20, <16 x float> undef, <16 x i32> zeroinitializer
  %22 = fmul <16 x float> %18, %21
  store <16 x float> %22, <16 x float>* %7, align 64, !tbaa !7
  %23 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %24 = load <16 x float>, <16 x float>* %9, align 64, !tbaa !7
  %25 = fmul <16 x float> %23, %24
  store <16 x float> %25, <16 x float>* %10, align 64, !tbaa !7
  %26 = load <16 x float>, <16 x float>* %10, align 64, !tbaa !7
  %27 = fptosi <16 x float> %26 to <16 x i32>
  %28 = trunc <16 x i32> %27 to <16 x i16>
  store <16 x i16> %28, <16 x i16>* %11, align 32, !tbaa !7
  %29 = load <16 x i16>, <16 x i16>* %4, align 32, !tbaa !7
  %30 = zext <16 x i16> %29 to <16 x i32>
  %31 = load <16 x i16>, <16 x i16>* %11, align 32, !tbaa !7
  %32 = zext <16 x i16> %31 to <16 x i32>
  %33 = load <16 x i16>, <16 x i16>* %5, align 32, !tbaa !7
  %34 = zext <16 x i16> %33 to <16 x i32>
  %35 = mul <16 x i32> %32, %34
  %36 = sub <16 x i32> %30, %35
  %37 = trunc <16 x i32> %36 to <16 x i16>
  call void @llvm.genx.vstore.v16i16.p0v16i16(<16 x i16> %37, <16 x i16>* %2)
  %38 = load <16 x i16>, <16 x i16>* %11, align 32, !tbaa !7
  ret <16 x i16> %38
}

; Function Attrs: noinline nounwind
define internal <32 x i16> @_Z24__cm_intrinsic_impl_udivu2CMvb32_tS_(<32 x i16>, <32 x i16>) #18 {
  %3 = alloca <32 x i16>, align 64
  %4 = alloca <32 x i16>, align 64
  %5 = alloca <32 x i16>, align 64
  %6 = alloca <32 x i16>, align 64
  store <32 x i16> %0, <32 x i16>* %3, align 64, !tbaa !7
  store <32 x i16> %1, <32 x i16>* %4, align 64, !tbaa !7
  %7 = load <32 x i16>, <32 x i16>* %3, align 64, !tbaa !7
  %8 = load <32 x i16>, <32 x i16>* %4, align 64, !tbaa !7
  %9 = call <32 x i16> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__tS1_S1_u2CMvrT__t(<32 x i16> %7, <32 x i16> %8, <32 x i16>* %5)
  store <32 x i16> %9, <32 x i16>* %6, align 64, !tbaa !7
  %10 = load <32 x i16>, <32 x i16>* %6, align 64, !tbaa !7
  ret <32 x i16> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <32 x i16> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__tS1_S1_u2CMvrT__t(<32 x i16>, <32 x i16>, <32 x i16>*) #12 comdat {
  %4 = alloca <32 x i16>, align 64
  %5 = alloca <32 x i16>, align 64
  %6 = alloca <32 x i16>*, align 64
  %7 = alloca <32 x float>, align 128
  %8 = alloca <32 x float>, align 128
  %9 = alloca <32 x float>, align 128
  %10 = alloca <32 x float>, align 128
  %11 = alloca <32 x i16>, align 64
  store <32 x i16> %0, <32 x i16>* %4, align 64, !tbaa !7
  store <32 x i16> %1, <32 x i16>* %5, align 64, !tbaa !7
  store <32 x i16>* %2, <32 x i16>** %6, align 64, !tbaa !7
  %12 = load <32 x i16>, <32 x i16>* %5, align 64, !tbaa !7
  %13 = uitofp <32 x i16> %12 to <32 x float>
  store <32 x float> %13, <32 x float>* %8, align 128, !tbaa !7
  %14 = load <32 x i16>, <32 x i16>* %4, align 64, !tbaa !7
  %15 = uitofp <32 x i16> %14 to <32 x float>
  store <32 x float> %15, <32 x float>* %7, align 128, !tbaa !7
  %16 = load <32 x float>, <32 x float>* %8, align 128, !tbaa !7
  %17 = fdiv <32 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %16
  store <32 x float> %17, <32 x float>* %9, align 128, !tbaa !7
  %18 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %19 = call float @_ZN7details16__impl_hex2floatEj(i32 1065353224)
  %20 = insertelement <32 x float> undef, float %19, i32 0
  %21 = shufflevector <32 x float> %20, <32 x float> undef, <32 x i32> zeroinitializer
  %22 = fmul <32 x float> %18, %21
  store <32 x float> %22, <32 x float>* %7, align 128, !tbaa !7
  %23 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %24 = load <32 x float>, <32 x float>* %9, align 128, !tbaa !7
  %25 = fmul <32 x float> %23, %24
  store <32 x float> %25, <32 x float>* %10, align 128, !tbaa !7
  %26 = load <32 x float>, <32 x float>* %10, align 128, !tbaa !7
  %27 = fptosi <32 x float> %26 to <32 x i32>
  %28 = trunc <32 x i32> %27 to <32 x i16>
  store <32 x i16> %28, <32 x i16>* %11, align 64, !tbaa !7
  %29 = load <32 x i16>, <32 x i16>* %4, align 64, !tbaa !7
  %30 = zext <32 x i16> %29 to <32 x i32>
  %31 = load <32 x i16>, <32 x i16>* %11, align 64, !tbaa !7
  %32 = zext <32 x i16> %31 to <32 x i32>
  %33 = load <32 x i16>, <32 x i16>* %5, align 64, !tbaa !7
  %34 = zext <32 x i16> %33 to <32 x i32>
  %35 = mul <32 x i32> %32, %34
  %36 = sub <32 x i32> %30, %35
  %37 = trunc <32 x i32> %36 to <32 x i16>
  call void @llvm.genx.vstore.v32i16.p0v32i16(<32 x i16> %37, <32 x i16>* %2)
  %38 = load <32 x i16>, <32 x i16>* %11, align 64, !tbaa !7
  ret <32 x i16> %38
}

; Function Attrs: noinline nounwind
define internal zeroext i16 @_Z24__cm_intrinsic_impl_uremtt(i16 zeroext, i16 zeroext) #22 {
  %3 = alloca i16, align 2
  %4 = alloca i16, align 2
  %5 = alloca <1 x i16>, align 2
  %6 = alloca <1 x i16>, align 2
  %7 = alloca <1 x i16>, align 2
  store i16 %0, i16* %3, align 2, !tbaa !11
  store i16 %1, i16* %4, align 2, !tbaa !11
  %8 = load i16, i16* %3, align 2, !tbaa !11
  %9 = insertelement <1 x i16> undef, i16 %8, i32 0
  %10 = shufflevector <1 x i16> %9, <1 x i16> undef, <1 x i32> zeroinitializer
  store <1 x i16> %10, <1 x i16>* %5, align 2, !tbaa !7
  %11 = load i16, i16* %4, align 2, !tbaa !11
  %12 = insertelement <1 x i16> undef, i16 %11, i32 0
  %13 = shufflevector <1 x i16> %12, <1 x i16> undef, <1 x i32> zeroinitializer
  store <1 x i16> %13, <1 x i16>* %6, align 2, !tbaa !7
  %14 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  %15 = load <1 x i16>, <1 x i16>* %6, align 2, !tbaa !7
  %16 = call <1 x i16> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__tS1_S1_u2CMvrT__t(<1 x i16> %14, <1 x i16> %15, <1 x i16>* %7)
  %17 = load <1 x i16>, <1 x i16>* %7, align 2, !tbaa !7
  %18 = call <1 x i16> @llvm.genx.rdregioni.v1i16.v1i16.i16(<1 x i16> %17, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %19 = extractelement <1 x i16> %18, i32 0
  ret i16 %19
}

; Function Attrs: noinline nounwind
define internal <1 x i16> @_Z24__cm_intrinsic_impl_uremu2CMvb1_tS_(<1 x i16>, <1 x i16>) #22 {
  %3 = alloca <1 x i16>, align 2
  %4 = alloca <1 x i16>, align 2
  %5 = alloca <1 x i16>, align 2
  store <1 x i16> %0, <1 x i16>* %3, align 2, !tbaa !7
  store <1 x i16> %1, <1 x i16>* %4, align 2, !tbaa !7
  %6 = load <1 x i16>, <1 x i16>* %3, align 2, !tbaa !7
  %7 = load <1 x i16>, <1 x i16>* %4, align 2, !tbaa !7
  %8 = call <1 x i16> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__tS1_S1_u2CMvrT__t(<1 x i16> %6, <1 x i16> %7, <1 x i16>* %5)
  %9 = load <1 x i16>, <1 x i16>* %5, align 2, !tbaa !7
  ret <1 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <2 x i16> @_Z24__cm_intrinsic_impl_uremu2CMvb2_tS_(<2 x i16>, <2 x i16>) #14 {
  %3 = alloca <2 x i16>, align 4
  %4 = alloca <2 x i16>, align 4
  %5 = alloca <2 x i16>, align 4
  store <2 x i16> %0, <2 x i16>* %3, align 4, !tbaa !7
  store <2 x i16> %1, <2 x i16>* %4, align 4, !tbaa !7
  %6 = load <2 x i16>, <2 x i16>* %3, align 4, !tbaa !7
  %7 = load <2 x i16>, <2 x i16>* %4, align 4, !tbaa !7
  %8 = call <2 x i16> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__tS1_S1_u2CMvrT__t(<2 x i16> %6, <2 x i16> %7, <2 x i16>* %5)
  %9 = load <2 x i16>, <2 x i16>* %5, align 4, !tbaa !7
  ret <2 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <4 x i16> @_Z24__cm_intrinsic_impl_uremu2CMvb4_tS_(<4 x i16>, <4 x i16>) #15 {
  %3 = alloca <4 x i16>, align 8
  %4 = alloca <4 x i16>, align 8
  %5 = alloca <4 x i16>, align 8
  store <4 x i16> %0, <4 x i16>* %3, align 8, !tbaa !7
  store <4 x i16> %1, <4 x i16>* %4, align 8, !tbaa !7
  %6 = load <4 x i16>, <4 x i16>* %3, align 8, !tbaa !7
  %7 = load <4 x i16>, <4 x i16>* %4, align 8, !tbaa !7
  %8 = call <4 x i16> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__tS1_S1_u2CMvrT__t(<4 x i16> %6, <4 x i16> %7, <4 x i16>* %5)
  %9 = load <4 x i16>, <4 x i16>* %5, align 8, !tbaa !7
  ret <4 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <8 x i16> @_Z24__cm_intrinsic_impl_uremu2CMvb8_tS_(<8 x i16>, <8 x i16>) #16 {
  %3 = alloca <8 x i16>, align 16
  %4 = alloca <8 x i16>, align 16
  %5 = alloca <8 x i16>, align 16
  store <8 x i16> %0, <8 x i16>* %3, align 16, !tbaa !7
  store <8 x i16> %1, <8 x i16>* %4, align 16, !tbaa !7
  %6 = load <8 x i16>, <8 x i16>* %3, align 16, !tbaa !7
  %7 = load <8 x i16>, <8 x i16>* %4, align 16, !tbaa !7
  %8 = call <8 x i16> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__tS1_S1_u2CMvrT__t(<8 x i16> %6, <8 x i16> %7, <8 x i16>* %5)
  %9 = load <8 x i16>, <8 x i16>* %5, align 16, !tbaa !7
  ret <8 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <16 x i16> @_Z24__cm_intrinsic_impl_uremu2CMvb16_tS_(<16 x i16>, <16 x i16>) #17 {
  %3 = alloca <16 x i16>, align 32
  %4 = alloca <16 x i16>, align 32
  %5 = alloca <16 x i16>, align 32
  store <16 x i16> %0, <16 x i16>* %3, align 32, !tbaa !7
  store <16 x i16> %1, <16 x i16>* %4, align 32, !tbaa !7
  %6 = load <16 x i16>, <16 x i16>* %3, align 32, !tbaa !7
  %7 = load <16 x i16>, <16 x i16>* %4, align 32, !tbaa !7
  %8 = call <16 x i16> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__tS1_S1_u2CMvrT__t(<16 x i16> %6, <16 x i16> %7, <16 x i16>* %5)
  %9 = load <16 x i16>, <16 x i16>* %5, align 32, !tbaa !7
  ret <16 x i16> %9
}

; Function Attrs: noinline nounwind
define internal <32 x i16> @_Z24__cm_intrinsic_impl_uremu2CMvb32_tS_(<32 x i16>, <32 x i16>) #18 {
  %3 = alloca <32 x i16>, align 64
  %4 = alloca <32 x i16>, align 64
  %5 = alloca <32 x i16>, align 64
  store <32 x i16> %0, <32 x i16>* %3, align 64, !tbaa !7
  store <32 x i16> %1, <32 x i16>* %4, align 64, !tbaa !7
  %6 = load <32 x i16>, <32 x i16>* %3, align 64, !tbaa !7
  %7 = load <32 x i16>, <32 x i16>* %4, align 64, !tbaa !7
  %8 = call <32 x i16> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__tS1_S1_u2CMvrT__t(<32 x i16> %6, <32 x i16> %7, <32 x i16>* %5)
  %9 = load <32 x i16>, <32 x i16>* %5, align 64, !tbaa !7
  ret <32 x i16> %9
}

; Function Attrs: noinline nounwind
define internal i32 @_Z24__cm_intrinsic_impl_udivjj(i32, i32) #14 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca <1 x i32>, align 4
  %6 = alloca <1 x i32>, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <1 x i32>, align 4
  store i32 %0, i32* %3, align 4, !tbaa !13
  store i32 %1, i32* %4, align 4, !tbaa !13
  %9 = load i32, i32* %3, align 4, !tbaa !13
  %10 = insertelement <1 x i32> undef, i32 %9, i32 0
  %11 = shufflevector <1 x i32> %10, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %11, <1 x i32>* %5, align 4, !tbaa !7
  %12 = load i32, i32* %4, align 4, !tbaa !13
  %13 = insertelement <1 x i32> undef, i32 %12, i32 0
  %14 = shufflevector <1 x i32> %13, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %14, <1 x i32>* %6, align 4, !tbaa !7
  %15 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %16 = load <1 x i32>, <1 x i32>* %6, align 4, !tbaa !7
  %17 = call <1 x i32> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__jS1_S1_u2CMvrT__j(<1 x i32> %15, <1 x i32> %16, <1 x i32>* %7)
  store <1 x i32> %17, <1 x i32>* %8, align 4, !tbaa !7
  %18 = load <1 x i32>, <1 x i32>* %8, align 4, !tbaa !7
  %19 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v1i32.i16(<1 x i32> %18, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %20 = extractelement <1 x i32> %19, i32 0
  ret i32 %20
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i32> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__jS1_S1_u2CMvrT__j(<1 x i32>, <1 x i32>, <1 x i32>*) #4 comdat {
  %4 = alloca <1 x i32>, align 4
  %5 = alloca <1 x i32>, align 4
  %6 = alloca <1 x i32>*, align 4
  %7 = alloca <1 x float>, align 4
  %8 = alloca <1 x float>, align 4
  %9 = alloca <1 x float>, align 4
  %10 = alloca <1 x float>, align 4
  %11 = alloca <1 x float>, align 4
  %12 = alloca <1 x float>, align 4
  %13 = alloca <1 x float>, align 4
  %14 = alloca <1 x float>, align 4
  %15 = alloca <1 x float>, align 4
  %16 = alloca <1 x i32>, align 4
  %17 = alloca <1 x i32>, align 4
  %18 = alloca <1 x i32>, align 4
  %19 = alloca <1 x i32>, align 4
  %20 = alloca <1 x i32>, align 4
  %21 = alloca <1 x i32>, align 4
  %22 = alloca <1 x i32>, align 4
  %23 = alloca <1 x i32>, align 4
  %24 = alloca <1 x i32>, align 4
  store <1 x i32> %0, <1 x i32>* %4, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %5, align 4, !tbaa !7
  store <1 x i32>* %2, <1 x i32>** %6, align 4, !tbaa !7
  %25 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %26 = uitofp <1 x i32> %25 to <1 x float>
  store <1 x float> %26, <1 x float>* %9, align 4, !tbaa !7
  %27 = load <1 x float>, <1 x float>* %9, align 4, !tbaa !7
  %28 = fptosi <1 x float> %27 to <1 x i32>
  store <1 x i32> %28, <1 x i32>* %17, align 4, !tbaa !7
  %29 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %30 = load <1 x i32>, <1 x i32>* %17, align 4, !tbaa !7
  %31 = sub <1 x i32> %29, %30
  store <1 x i32> %31, <1 x i32>* %19, align 4, !tbaa !7
  %32 = load <1 x i32>, <1 x i32>* %19, align 4, !tbaa !7
  %33 = uitofp <1 x i32> %32 to <1 x float>
  store <1 x float> %33, <1 x float>* %10, align 4, !tbaa !7
  %34 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %35 = uitofp <1 x i32> %34 to <1 x float>
  store <1 x float> %35, <1 x float>* %7, align 4, !tbaa !7
  %36 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %37 = fptosi <1 x float> %36 to <1 x i32>
  store <1 x i32> %37, <1 x i32>* %16, align 4, !tbaa !7
  %38 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %39 = load <1 x i32>, <1 x i32>* %16, align 4, !tbaa !7
  %40 = sub <1 x i32> %38, %39
  store <1 x i32> %40, <1 x i32>* %18, align 4, !tbaa !7
  %41 = load <1 x i32>, <1 x i32>* %18, align 4, !tbaa !7
  %42 = uitofp <1 x i32> %41 to <1 x float>
  store <1 x float> %42, <1 x float>* %8, align 4, !tbaa !7
  %43 = load <1 x float>, <1 x float>* %9, align 4, !tbaa !7
  %44 = fdiv <1 x float> <float 1.000000e+00>, %43
  store <1 x float> %44, <1 x float>* %11, align 4, !tbaa !7
  %45 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %46 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %47 = insertelement <1 x float> undef, float %46, i32 0
  %48 = shufflevector <1 x float> %47, <1 x float> undef, <1 x i32> zeroinitializer
  %49 = fmul <1 x float> %45, %48
  %50 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %51 = fadd <1 x float> %50, %49
  store <1 x float> %51, <1 x float>* %11, align 4, !tbaa !7
  %52 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %53 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %54 = fmul <1 x float> %52, %53
  store <1 x float> %54, <1 x float>* %12, align 4, !tbaa !7
  %55 = load <1 x float>, <1 x float>* %12, align 4, !tbaa !7
  %56 = fptosi <1 x float> %55 to <1 x i32>
  store <1 x i32> %56, <1 x i32>* %20, align 4, !tbaa !7
  %57 = load <1 x i32>, <1 x i32>* %20, align 4, !tbaa !7
  %58 = uitofp <1 x i32> %57 to <1 x float>
  store <1 x float> %58, <1 x float>* %12, align 4, !tbaa !7
  %59 = load <1 x float>, <1 x float>* %7, align 4, !tbaa !7
  %60 = load <1 x float>, <1 x float>* %9, align 4, !tbaa !7
  %61 = load <1 x float>, <1 x float>* %12, align 4, !tbaa !7
  %62 = fmul <1 x float> %60, %61
  %63 = fsub <1 x float> %59, %62
  store <1 x float> %63, <1 x float>* %14, align 4, !tbaa !7
  %64 = load <1 x float>, <1 x float>* %8, align 4, !tbaa !7
  %65 = load <1 x float>, <1 x float>* %10, align 4, !tbaa !7
  %66 = load <1 x float>, <1 x float>* %12, align 4, !tbaa !7
  %67 = fmul <1 x float> %65, %66
  %68 = fsub <1 x float> %64, %67
  store <1 x float> %68, <1 x float>* %15, align 4, !tbaa !7
  %69 = load <1 x float>, <1 x float>* %14, align 4, !tbaa !7
  %70 = load <1 x float>, <1 x float>* %15, align 4, !tbaa !7
  %71 = fadd <1 x float> %69, %70
  store <1 x float> %71, <1 x float>* %15, align 4, !tbaa !7
  %72 = load <1 x float>, <1 x float>* %11, align 4, !tbaa !7
  %73 = load <1 x float>, <1 x float>* %15, align 4, !tbaa !7
  %74 = fmul <1 x float> %72, %73
  store <1 x float> %74, <1 x float>* %13, align 4, !tbaa !7
  %75 = load <1 x float>, <1 x float>* %13, align 4, !tbaa !7
  %76 = fptosi <1 x float> %75 to <1 x i32>
  store <1 x i32> %76, <1 x i32>* %21, align 4, !tbaa !7
  %77 = load <1 x i32>, <1 x i32>* %20, align 4, !tbaa !7
  %78 = load <1 x i32>, <1 x i32>* %21, align 4, !tbaa !7
  %79 = add <1 x i32> %77, %78
  store <1 x i32> %79, <1 x i32>* %22, align 4, !tbaa !7
  %80 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %81 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %82 = load <1 x i32>, <1 x i32>* %22, align 4, !tbaa !7
  %83 = mul <1 x i32> %81, %82
  %84 = sub <1 x i32> %80, %83
  store <1 x i32> %84, <1 x i32>* %23, align 4, !tbaa !7
  %85 = load <1 x i32>, <1 x i32>* %23, align 4, !tbaa !7
  %86 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %87 = icmp uge <1 x i32> %85, %86
  %88 = zext <1 x i1> %87 to <1 x i16>
  %89 = trunc <1 x i16> %88 to <1 x i1>
  %90 = select <1 x i1> %89, <1 x i32> <i32 -1>, <1 x i32> zeroinitializer
  store <1 x i32> %90, <1 x i32>* %24, align 4, !tbaa !7
  %91 = load <1 x i32>, <1 x i32>* %24, align 4, !tbaa !7
  %92 = and <1 x i32> %91, <i32 1>
  %93 = load <1 x i32>, <1 x i32>* %22, align 4, !tbaa !7
  %94 = add <1 x i32> %93, %92
  store <1 x i32> %94, <1 x i32>* %22, align 4, !tbaa !7
  %95 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %96 = load <1 x i32>, <1 x i32>* %24, align 4, !tbaa !7
  %97 = and <1 x i32> %95, %96
  %98 = load <1 x i32>, <1 x i32>* %23, align 4, !tbaa !7
  %99 = sub <1 x i32> %98, %97
  store <1 x i32> %99, <1 x i32>* %23, align 4, !tbaa !7
  %100 = load <1 x i32>, <1 x i32>* %23, align 4, !tbaa !7
  store <1 x i32> %100, <1 x i32>* %2
  %101 = load <1 x i32>, <1 x i32>* %22, align 4, !tbaa !7
  ret <1 x i32> %101
}

; Function Attrs: noinline nounwind
define internal <1 x i32> @_Z24__cm_intrinsic_impl_udivu2CMvb1_jS_(<1 x i32>, <1 x i32>) #14 {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x i32>, align 4
  %5 = alloca <1 x i32>, align 4
  %6 = alloca <1 x i32>, align 4
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %4, align 4, !tbaa !7
  %7 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %8 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %9 = call <1 x i32> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__jS1_S1_u2CMvrT__j(<1 x i32> %7, <1 x i32> %8, <1 x i32>* %5)
  store <1 x i32> %9, <1 x i32>* %6, align 4, !tbaa !7
  %10 = load <1 x i32>, <1 x i32>* %6, align 4, !tbaa !7
  ret <1 x i32> %10
}

; Function Attrs: noinline nounwind
define internal <2 x i32> @_Z24__cm_intrinsic_impl_udivu2CMvb2_jS_(<2 x i32>, <2 x i32>) #15 {
  %3 = alloca <2 x i32>, align 8
  %4 = alloca <2 x i32>, align 8
  %5 = alloca <2 x i32>, align 8
  %6 = alloca <2 x i32>, align 8
  store <2 x i32> %0, <2 x i32>* %3, align 8, !tbaa !7
  store <2 x i32> %1, <2 x i32>* %4, align 8, !tbaa !7
  %7 = load <2 x i32>, <2 x i32>* %3, align 8, !tbaa !7
  %8 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %9 = call <2 x i32> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__jS1_S1_u2CMvrT__j(<2 x i32> %7, <2 x i32> %8, <2 x i32>* %5)
  store <2 x i32> %9, <2 x i32>* %6, align 8, !tbaa !7
  %10 = load <2 x i32>, <2 x i32>* %6, align 8, !tbaa !7
  ret <2 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <2 x i32> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__jS1_S1_u2CMvrT__j(<2 x i32>, <2 x i32>, <2 x i32>*) #6 comdat {
  %4 = alloca <2 x i32>, align 8
  %5 = alloca <2 x i32>, align 8
  %6 = alloca <2 x i32>*, align 8
  %7 = alloca <2 x float>, align 8
  %8 = alloca <2 x float>, align 8
  %9 = alloca <2 x float>, align 8
  %10 = alloca <2 x float>, align 8
  %11 = alloca <2 x float>, align 8
  %12 = alloca <2 x float>, align 8
  %13 = alloca <2 x float>, align 8
  %14 = alloca <2 x float>, align 8
  %15 = alloca <2 x float>, align 8
  %16 = alloca <2 x i32>, align 8
  %17 = alloca <2 x i32>, align 8
  %18 = alloca <2 x i32>, align 8
  %19 = alloca <2 x i32>, align 8
  %20 = alloca <2 x i32>, align 8
  %21 = alloca <2 x i32>, align 8
  %22 = alloca <2 x i32>, align 8
  %23 = alloca <2 x i32>, align 8
  %24 = alloca <2 x i32>, align 8
  store <2 x i32> %0, <2 x i32>* %4, align 8, !tbaa !7
  store <2 x i32> %1, <2 x i32>* %5, align 8, !tbaa !7
  store <2 x i32>* %2, <2 x i32>** %6, align 8, !tbaa !7
  %25 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  %26 = uitofp <2 x i32> %25 to <2 x float>
  store <2 x float> %26, <2 x float>* %9, align 8, !tbaa !7
  %27 = load <2 x float>, <2 x float>* %9, align 8, !tbaa !7
  %28 = fptosi <2 x float> %27 to <2 x i32>
  store <2 x i32> %28, <2 x i32>* %17, align 8, !tbaa !7
  %29 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  %30 = load <2 x i32>, <2 x i32>* %17, align 8, !tbaa !7
  %31 = sub <2 x i32> %29, %30
  store <2 x i32> %31, <2 x i32>* %19, align 8, !tbaa !7
  %32 = load <2 x i32>, <2 x i32>* %19, align 8, !tbaa !7
  %33 = uitofp <2 x i32> %32 to <2 x float>
  store <2 x float> %33, <2 x float>* %10, align 8, !tbaa !7
  %34 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %35 = uitofp <2 x i32> %34 to <2 x float>
  store <2 x float> %35, <2 x float>* %7, align 8, !tbaa !7
  %36 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %37 = fptosi <2 x float> %36 to <2 x i32>
  store <2 x i32> %37, <2 x i32>* %16, align 8, !tbaa !7
  %38 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %39 = load <2 x i32>, <2 x i32>* %16, align 8, !tbaa !7
  %40 = sub <2 x i32> %38, %39
  store <2 x i32> %40, <2 x i32>* %18, align 8, !tbaa !7
  %41 = load <2 x i32>, <2 x i32>* %18, align 8, !tbaa !7
  %42 = uitofp <2 x i32> %41 to <2 x float>
  store <2 x float> %42, <2 x float>* %8, align 8, !tbaa !7
  %43 = load <2 x float>, <2 x float>* %9, align 8, !tbaa !7
  %44 = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %43
  store <2 x float> %44, <2 x float>* %11, align 8, !tbaa !7
  %45 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %46 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %47 = insertelement <2 x float> undef, float %46, i32 0
  %48 = shufflevector <2 x float> %47, <2 x float> undef, <2 x i32> zeroinitializer
  %49 = fmul <2 x float> %45, %48
  %50 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %51 = fadd <2 x float> %50, %49
  store <2 x float> %51, <2 x float>* %11, align 8, !tbaa !7
  %52 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %53 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %54 = fmul <2 x float> %52, %53
  store <2 x float> %54, <2 x float>* %12, align 8, !tbaa !7
  %55 = load <2 x float>, <2 x float>* %12, align 8, !tbaa !7
  %56 = fptosi <2 x float> %55 to <2 x i32>
  store <2 x i32> %56, <2 x i32>* %20, align 8, !tbaa !7
  %57 = load <2 x i32>, <2 x i32>* %20, align 8, !tbaa !7
  %58 = uitofp <2 x i32> %57 to <2 x float>
  store <2 x float> %58, <2 x float>* %12, align 8, !tbaa !7
  %59 = load <2 x float>, <2 x float>* %7, align 8, !tbaa !7
  %60 = load <2 x float>, <2 x float>* %9, align 8, !tbaa !7
  %61 = load <2 x float>, <2 x float>* %12, align 8, !tbaa !7
  %62 = fmul <2 x float> %60, %61
  %63 = fsub <2 x float> %59, %62
  store <2 x float> %63, <2 x float>* %14, align 8, !tbaa !7
  %64 = load <2 x float>, <2 x float>* %8, align 8, !tbaa !7
  %65 = load <2 x float>, <2 x float>* %10, align 8, !tbaa !7
  %66 = load <2 x float>, <2 x float>* %12, align 8, !tbaa !7
  %67 = fmul <2 x float> %65, %66
  %68 = fsub <2 x float> %64, %67
  store <2 x float> %68, <2 x float>* %15, align 8, !tbaa !7
  %69 = load <2 x float>, <2 x float>* %14, align 8, !tbaa !7
  %70 = load <2 x float>, <2 x float>* %15, align 8, !tbaa !7
  %71 = fadd <2 x float> %69, %70
  store <2 x float> %71, <2 x float>* %15, align 8, !tbaa !7
  %72 = load <2 x float>, <2 x float>* %11, align 8, !tbaa !7
  %73 = load <2 x float>, <2 x float>* %15, align 8, !tbaa !7
  %74 = fmul <2 x float> %72, %73
  store <2 x float> %74, <2 x float>* %13, align 8, !tbaa !7
  %75 = load <2 x float>, <2 x float>* %13, align 8, !tbaa !7
  %76 = fptosi <2 x float> %75 to <2 x i32>
  store <2 x i32> %76, <2 x i32>* %21, align 8, !tbaa !7
  %77 = load <2 x i32>, <2 x i32>* %20, align 8, !tbaa !7
  %78 = load <2 x i32>, <2 x i32>* %21, align 8, !tbaa !7
  %79 = add <2 x i32> %77, %78
  store <2 x i32> %79, <2 x i32>* %22, align 8, !tbaa !7
  %80 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %81 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  %82 = load <2 x i32>, <2 x i32>* %22, align 8, !tbaa !7
  %83 = mul <2 x i32> %81, %82
  %84 = sub <2 x i32> %80, %83
  store <2 x i32> %84, <2 x i32>* %23, align 8, !tbaa !7
  %85 = load <2 x i32>, <2 x i32>* %23, align 8, !tbaa !7
  %86 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  %87 = icmp uge <2 x i32> %85, %86
  %88 = zext <2 x i1> %87 to <2 x i16>
  %89 = trunc <2 x i16> %88 to <2 x i1>
  %90 = select <2 x i1> %89, <2 x i32> <i32 -1, i32 -1>, <2 x i32> zeroinitializer
  store <2 x i32> %90, <2 x i32>* %24, align 8, !tbaa !7
  %91 = load <2 x i32>, <2 x i32>* %24, align 8, !tbaa !7
  %92 = and <2 x i32> %91, <i32 1, i32 1>
  %93 = load <2 x i32>, <2 x i32>* %22, align 8, !tbaa !7
  %94 = add <2 x i32> %93, %92
  store <2 x i32> %94, <2 x i32>* %22, align 8, !tbaa !7
  %95 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  %96 = load <2 x i32>, <2 x i32>* %24, align 8, !tbaa !7
  %97 = and <2 x i32> %95, %96
  %98 = load <2 x i32>, <2 x i32>* %23, align 8, !tbaa !7
  %99 = sub <2 x i32> %98, %97
  store <2 x i32> %99, <2 x i32>* %23, align 8, !tbaa !7
  %100 = load <2 x i32>, <2 x i32>* %23, align 8, !tbaa !7
  call void @llvm.genx.vstore.v2i32.p0v2i32(<2 x i32> %100, <2 x i32>* %2)
  %101 = load <2 x i32>, <2 x i32>* %22, align 8, !tbaa !7
  ret <2 x i32> %101
}

; Function Attrs: noinline nounwind
define internal <4 x i32> @_Z24__cm_intrinsic_impl_udivu2CMvb4_jS_(<4 x i32>, <4 x i32>) #16 {
  %3 = alloca <4 x i32>, align 16
  %4 = alloca <4 x i32>, align 16
  %5 = alloca <4 x i32>, align 16
  %6 = alloca <4 x i32>, align 16
  store <4 x i32> %0, <4 x i32>* %3, align 16, !tbaa !7
  store <4 x i32> %1, <4 x i32>* %4, align 16, !tbaa !7
  %7 = load <4 x i32>, <4 x i32>* %3, align 16, !tbaa !7
  %8 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %9 = call <4 x i32> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__jS1_S1_u2CMvrT__j(<4 x i32> %7, <4 x i32> %8, <4 x i32>* %5)
  store <4 x i32> %9, <4 x i32>* %6, align 16, !tbaa !7
  %10 = load <4 x i32>, <4 x i32>* %6, align 16, !tbaa !7
  ret <4 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <4 x i32> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__jS1_S1_u2CMvrT__j(<4 x i32>, <4 x i32>, <4 x i32>*) #8 comdat {
  %4 = alloca <4 x i32>, align 16
  %5 = alloca <4 x i32>, align 16
  %6 = alloca <4 x i32>*, align 16
  %7 = alloca <4 x float>, align 16
  %8 = alloca <4 x float>, align 16
  %9 = alloca <4 x float>, align 16
  %10 = alloca <4 x float>, align 16
  %11 = alloca <4 x float>, align 16
  %12 = alloca <4 x float>, align 16
  %13 = alloca <4 x float>, align 16
  %14 = alloca <4 x float>, align 16
  %15 = alloca <4 x float>, align 16
  %16 = alloca <4 x i32>, align 16
  %17 = alloca <4 x i32>, align 16
  %18 = alloca <4 x i32>, align 16
  %19 = alloca <4 x i32>, align 16
  %20 = alloca <4 x i32>, align 16
  %21 = alloca <4 x i32>, align 16
  %22 = alloca <4 x i32>, align 16
  %23 = alloca <4 x i32>, align 16
  %24 = alloca <4 x i32>, align 16
  store <4 x i32> %0, <4 x i32>* %4, align 16, !tbaa !7
  store <4 x i32> %1, <4 x i32>* %5, align 16, !tbaa !7
  store <4 x i32>* %2, <4 x i32>** %6, align 16, !tbaa !7
  %25 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  %26 = uitofp <4 x i32> %25 to <4 x float>
  store <4 x float> %26, <4 x float>* %9, align 16, !tbaa !7
  %27 = load <4 x float>, <4 x float>* %9, align 16, !tbaa !7
  %28 = fptosi <4 x float> %27 to <4 x i32>
  store <4 x i32> %28, <4 x i32>* %17, align 16, !tbaa !7
  %29 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  %30 = load <4 x i32>, <4 x i32>* %17, align 16, !tbaa !7
  %31 = sub <4 x i32> %29, %30
  store <4 x i32> %31, <4 x i32>* %19, align 16, !tbaa !7
  %32 = load <4 x i32>, <4 x i32>* %19, align 16, !tbaa !7
  %33 = uitofp <4 x i32> %32 to <4 x float>
  store <4 x float> %33, <4 x float>* %10, align 16, !tbaa !7
  %34 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %35 = uitofp <4 x i32> %34 to <4 x float>
  store <4 x float> %35, <4 x float>* %7, align 16, !tbaa !7
  %36 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %37 = fptosi <4 x float> %36 to <4 x i32>
  store <4 x i32> %37, <4 x i32>* %16, align 16, !tbaa !7
  %38 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %39 = load <4 x i32>, <4 x i32>* %16, align 16, !tbaa !7
  %40 = sub <4 x i32> %38, %39
  store <4 x i32> %40, <4 x i32>* %18, align 16, !tbaa !7
  %41 = load <4 x i32>, <4 x i32>* %18, align 16, !tbaa !7
  %42 = uitofp <4 x i32> %41 to <4 x float>
  store <4 x float> %42, <4 x float>* %8, align 16, !tbaa !7
  %43 = load <4 x float>, <4 x float>* %9, align 16, !tbaa !7
  %44 = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %43
  store <4 x float> %44, <4 x float>* %11, align 16, !tbaa !7
  %45 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %46 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %47 = insertelement <4 x float> undef, float %46, i32 0
  %48 = shufflevector <4 x float> %47, <4 x float> undef, <4 x i32> zeroinitializer
  %49 = fmul <4 x float> %45, %48
  %50 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %51 = fadd <4 x float> %50, %49
  store <4 x float> %51, <4 x float>* %11, align 16, !tbaa !7
  %52 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %53 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %54 = fmul <4 x float> %52, %53
  store <4 x float> %54, <4 x float>* %12, align 16, !tbaa !7
  %55 = load <4 x float>, <4 x float>* %12, align 16, !tbaa !7
  %56 = fptosi <4 x float> %55 to <4 x i32>
  store <4 x i32> %56, <4 x i32>* %20, align 16, !tbaa !7
  %57 = load <4 x i32>, <4 x i32>* %20, align 16, !tbaa !7
  %58 = uitofp <4 x i32> %57 to <4 x float>
  store <4 x float> %58, <4 x float>* %12, align 16, !tbaa !7
  %59 = load <4 x float>, <4 x float>* %7, align 16, !tbaa !7
  %60 = load <4 x float>, <4 x float>* %9, align 16, !tbaa !7
  %61 = load <4 x float>, <4 x float>* %12, align 16, !tbaa !7
  %62 = fmul <4 x float> %60, %61
  %63 = fsub <4 x float> %59, %62
  store <4 x float> %63, <4 x float>* %14, align 16, !tbaa !7
  %64 = load <4 x float>, <4 x float>* %8, align 16, !tbaa !7
  %65 = load <4 x float>, <4 x float>* %10, align 16, !tbaa !7
  %66 = load <4 x float>, <4 x float>* %12, align 16, !tbaa !7
  %67 = fmul <4 x float> %65, %66
  %68 = fsub <4 x float> %64, %67
  store <4 x float> %68, <4 x float>* %15, align 16, !tbaa !7
  %69 = load <4 x float>, <4 x float>* %14, align 16, !tbaa !7
  %70 = load <4 x float>, <4 x float>* %15, align 16, !tbaa !7
  %71 = fadd <4 x float> %69, %70
  store <4 x float> %71, <4 x float>* %15, align 16, !tbaa !7
  %72 = load <4 x float>, <4 x float>* %11, align 16, !tbaa !7
  %73 = load <4 x float>, <4 x float>* %15, align 16, !tbaa !7
  %74 = fmul <4 x float> %72, %73
  store <4 x float> %74, <4 x float>* %13, align 16, !tbaa !7
  %75 = load <4 x float>, <4 x float>* %13, align 16, !tbaa !7
  %76 = fptosi <4 x float> %75 to <4 x i32>
  store <4 x i32> %76, <4 x i32>* %21, align 16, !tbaa !7
  %77 = load <4 x i32>, <4 x i32>* %20, align 16, !tbaa !7
  %78 = load <4 x i32>, <4 x i32>* %21, align 16, !tbaa !7
  %79 = add <4 x i32> %77, %78
  store <4 x i32> %79, <4 x i32>* %22, align 16, !tbaa !7
  %80 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %81 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  %82 = load <4 x i32>, <4 x i32>* %22, align 16, !tbaa !7
  %83 = mul <4 x i32> %81, %82
  %84 = sub <4 x i32> %80, %83
  store <4 x i32> %84, <4 x i32>* %23, align 16, !tbaa !7
  %85 = load <4 x i32>, <4 x i32>* %23, align 16, !tbaa !7
  %86 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  %87 = icmp uge <4 x i32> %85, %86
  %88 = zext <4 x i1> %87 to <4 x i16>
  %89 = trunc <4 x i16> %88 to <4 x i1>
  %90 = select <4 x i1> %89, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer
  store <4 x i32> %90, <4 x i32>* %24, align 16, !tbaa !7
  %91 = load <4 x i32>, <4 x i32>* %24, align 16, !tbaa !7
  %92 = and <4 x i32> %91, <i32 1, i32 1, i32 1, i32 1>
  %93 = load <4 x i32>, <4 x i32>* %22, align 16, !tbaa !7
  %94 = add <4 x i32> %93, %92
  store <4 x i32> %94, <4 x i32>* %22, align 16, !tbaa !7
  %95 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  %96 = load <4 x i32>, <4 x i32>* %24, align 16, !tbaa !7
  %97 = and <4 x i32> %95, %96
  %98 = load <4 x i32>, <4 x i32>* %23, align 16, !tbaa !7
  %99 = sub <4 x i32> %98, %97
  store <4 x i32> %99, <4 x i32>* %23, align 16, !tbaa !7
  %100 = load <4 x i32>, <4 x i32>* %23, align 16, !tbaa !7
  call void @llvm.genx.vstore.v4i32.p0v4i32(<4 x i32> %100, <4 x i32>* %2)
  %101 = load <4 x i32>, <4 x i32>* %22, align 16, !tbaa !7
  ret <4 x i32> %101
}

; Function Attrs: noinline nounwind
define dso_local <8 x i32> @_Z24__cm_intrinsic_impl_udivu2CMvb8_jS_(<8 x i32>, <8 x i32>) #17 {
  %3 = alloca <8 x i32>, align 32
  %4 = alloca <8 x i32>, align 32
  %5 = alloca <8 x i32>, align 32
  %6 = alloca <8 x i32>, align 32
  store <8 x i32> %0, <8 x i32>* %3, align 32, !tbaa !7
  store <8 x i32> %1, <8 x i32>* %4, align 32, !tbaa !7
  %7 = load <8 x i32>, <8 x i32>* %3, align 32, !tbaa !7
  %8 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %9 = call <8 x i32> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__jS1_S1_u2CMvrT__j(<8 x i32> %7, <8 x i32> %8, <8 x i32>* %5)
  store <8 x i32> %9, <8 x i32>* %6, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %6, align 32, !tbaa !7
  ret <8 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <8 x i32> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__jS1_S1_u2CMvrT__j(<8 x i32>, <8 x i32>, <8 x i32>*) #10 comdat {
  %4 = alloca <8 x i32>, align 32
  %5 = alloca <8 x i32>, align 32
  %6 = alloca <8 x i32>*, align 32
  %7 = alloca <8 x float>, align 32
  %8 = alloca <8 x float>, align 32
  %9 = alloca <8 x float>, align 32
  %10 = alloca <8 x float>, align 32
  %11 = alloca <8 x float>, align 32
  %12 = alloca <8 x float>, align 32
  %13 = alloca <8 x float>, align 32
  %14 = alloca <8 x float>, align 32
  %15 = alloca <8 x float>, align 32
  %16 = alloca <8 x i32>, align 32
  %17 = alloca <8 x i32>, align 32
  %18 = alloca <8 x i32>, align 32
  %19 = alloca <8 x i32>, align 32
  %20 = alloca <8 x i32>, align 32
  %21 = alloca <8 x i32>, align 32
  %22 = alloca <8 x i32>, align 32
  %23 = alloca <8 x i32>, align 32
  %24 = alloca <8 x i32>, align 32
  store <8 x i32> %0, <8 x i32>* %4, align 32, !tbaa !7
  store <8 x i32> %1, <8 x i32>* %5, align 32, !tbaa !7
  store <8 x i32>* %2, <8 x i32>** %6, align 32, !tbaa !7
  %25 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %26 = uitofp <8 x i32> %25 to <8 x float>
  store <8 x float> %26, <8 x float>* %9, align 32, !tbaa !7
  %27 = load <8 x float>, <8 x float>* %9, align 32, !tbaa !7
  %28 = fptosi <8 x float> %27 to <8 x i32>
  store <8 x i32> %28, <8 x i32>* %17, align 32, !tbaa !7
  %29 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %30 = load <8 x i32>, <8 x i32>* %17, align 32, !tbaa !7
  %31 = sub <8 x i32> %29, %30
  store <8 x i32> %31, <8 x i32>* %19, align 32, !tbaa !7
  %32 = load <8 x i32>, <8 x i32>* %19, align 32, !tbaa !7
  %33 = uitofp <8 x i32> %32 to <8 x float>
  store <8 x float> %33, <8 x float>* %10, align 32, !tbaa !7
  %34 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %35 = uitofp <8 x i32> %34 to <8 x float>
  store <8 x float> %35, <8 x float>* %7, align 32, !tbaa !7
  %36 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %37 = fptosi <8 x float> %36 to <8 x i32>
  store <8 x i32> %37, <8 x i32>* %16, align 32, !tbaa !7
  %38 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %39 = load <8 x i32>, <8 x i32>* %16, align 32, !tbaa !7
  %40 = sub <8 x i32> %38, %39
  store <8 x i32> %40, <8 x i32>* %18, align 32, !tbaa !7
  %41 = load <8 x i32>, <8 x i32>* %18, align 32, !tbaa !7
  %42 = uitofp <8 x i32> %41 to <8 x float>
  store <8 x float> %42, <8 x float>* %8, align 32, !tbaa !7
  %43 = load <8 x float>, <8 x float>* %9, align 32, !tbaa !7
  %44 = fdiv <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %43
  store <8 x float> %44, <8 x float>* %11, align 32, !tbaa !7
  %45 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %46 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %47 = insertelement <8 x float> undef, float %46, i32 0
  %48 = shufflevector <8 x float> %47, <8 x float> undef, <8 x i32> zeroinitializer
  %49 = fmul <8 x float> %45, %48
  %50 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %51 = fadd <8 x float> %50, %49
  store <8 x float> %51, <8 x float>* %11, align 32, !tbaa !7
  %52 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %53 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %54 = fmul <8 x float> %52, %53
  store <8 x float> %54, <8 x float>* %12, align 32, !tbaa !7
  %55 = load <8 x float>, <8 x float>* %12, align 32, !tbaa !7
  %56 = fptosi <8 x float> %55 to <8 x i32>
  store <8 x i32> %56, <8 x i32>* %20, align 32, !tbaa !7
  %57 = load <8 x i32>, <8 x i32>* %20, align 32, !tbaa !7
  %58 = uitofp <8 x i32> %57 to <8 x float>
  store <8 x float> %58, <8 x float>* %12, align 32, !tbaa !7
  %59 = load <8 x float>, <8 x float>* %7, align 32, !tbaa !7
  %60 = load <8 x float>, <8 x float>* %9, align 32, !tbaa !7
  %61 = load <8 x float>, <8 x float>* %12, align 32, !tbaa !7
  %62 = fmul <8 x float> %60, %61
  %63 = fsub <8 x float> %59, %62
  store <8 x float> %63, <8 x float>* %14, align 32, !tbaa !7
  %64 = load <8 x float>, <8 x float>* %8, align 32, !tbaa !7
  %65 = load <8 x float>, <8 x float>* %10, align 32, !tbaa !7
  %66 = load <8 x float>, <8 x float>* %12, align 32, !tbaa !7
  %67 = fmul <8 x float> %65, %66
  %68 = fsub <8 x float> %64, %67
  store <8 x float> %68, <8 x float>* %15, align 32, !tbaa !7
  %69 = load <8 x float>, <8 x float>* %14, align 32, !tbaa !7
  %70 = load <8 x float>, <8 x float>* %15, align 32, !tbaa !7
  %71 = fadd <8 x float> %69, %70
  store <8 x float> %71, <8 x float>* %15, align 32, !tbaa !7
  %72 = load <8 x float>, <8 x float>* %11, align 32, !tbaa !7
  %73 = load <8 x float>, <8 x float>* %15, align 32, !tbaa !7
  %74 = fmul <8 x float> %72, %73
  store <8 x float> %74, <8 x float>* %13, align 32, !tbaa !7
  %75 = load <8 x float>, <8 x float>* %13, align 32, !tbaa !7
  %76 = fptosi <8 x float> %75 to <8 x i32>
  store <8 x i32> %76, <8 x i32>* %21, align 32, !tbaa !7
  %77 = load <8 x i32>, <8 x i32>* %20, align 32, !tbaa !7
  %78 = load <8 x i32>, <8 x i32>* %21, align 32, !tbaa !7
  %79 = add <8 x i32> %77, %78
  store <8 x i32> %79, <8 x i32>* %22, align 32, !tbaa !7
  %80 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %81 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %82 = load <8 x i32>, <8 x i32>* %22, align 32, !tbaa !7
  %83 = mul <8 x i32> %81, %82
  %84 = sub <8 x i32> %80, %83
  store <8 x i32> %84, <8 x i32>* %23, align 32, !tbaa !7
  %85 = load <8 x i32>, <8 x i32>* %23, align 32, !tbaa !7
  %86 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %87 = icmp uge <8 x i32> %85, %86
  %88 = zext <8 x i1> %87 to <8 x i16>
  %89 = trunc <8 x i16> %88 to <8 x i1>
  %90 = select <8 x i1> %89, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer
  store <8 x i32> %90, <8 x i32>* %24, align 32, !tbaa !7
  %91 = load <8 x i32>, <8 x i32>* %24, align 32, !tbaa !7
  %92 = and <8 x i32> %91, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %93 = load <8 x i32>, <8 x i32>* %22, align 32, !tbaa !7
  %94 = add <8 x i32> %93, %92
  store <8 x i32> %94, <8 x i32>* %22, align 32, !tbaa !7
  %95 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %96 = load <8 x i32>, <8 x i32>* %24, align 32, !tbaa !7
  %97 = and <8 x i32> %95, %96
  %98 = load <8 x i32>, <8 x i32>* %23, align 32, !tbaa !7
  %99 = sub <8 x i32> %98, %97
  store <8 x i32> %99, <8 x i32>* %23, align 32, !tbaa !7
  %100 = load <8 x i32>, <8 x i32>* %23, align 32, !tbaa !7
  call void @llvm.genx.vstore.v8i32.p0v8i32(<8 x i32> %100, <8 x i32>* %2)
  %101 = load <8 x i32>, <8 x i32>* %22, align 32, !tbaa !7
  ret <8 x i32> %101
}

; Function Attrs: noinline nounwind
define dso_local <16 x i32> @_Z24__cm_intrinsic_impl_udivu2CMvb16_jS_(<16 x i32>, <16 x i32>) #18 {
  %3 = alloca <16 x i32>, align 64
  %4 = alloca <16 x i32>, align 64
  %5 = alloca <16 x i32>, align 64
  %6 = alloca <16 x i32>, align 64
  store <16 x i32> %0, <16 x i32>* %3, align 64, !tbaa !7
  store <16 x i32> %1, <16 x i32>* %4, align 64, !tbaa !7
  %7 = load <16 x i32>, <16 x i32>* %3, align 64, !tbaa !7
  %8 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %9 = call <16 x i32> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__jS1_S1_u2CMvrT__j(<16 x i32> %7, <16 x i32> %8, <16 x i32>* %5)
  store <16 x i32> %9, <16 x i32>* %6, align 64, !tbaa !7
  %10 = load <16 x i32>, <16 x i32>* %6, align 64, !tbaa !7
  ret <16 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <16 x i32> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__jS1_S1_u2CMvrT__j(<16 x i32>, <16 x i32>, <16 x i32>*) #12 comdat {
  %4 = alloca <16 x i32>, align 64
  %5 = alloca <16 x i32>, align 64
  %6 = alloca <16 x i32>*, align 64
  %7 = alloca <16 x float>, align 64
  %8 = alloca <16 x float>, align 64
  %9 = alloca <16 x float>, align 64
  %10 = alloca <16 x float>, align 64
  %11 = alloca <16 x float>, align 64
  %12 = alloca <16 x float>, align 64
  %13 = alloca <16 x float>, align 64
  %14 = alloca <16 x float>, align 64
  %15 = alloca <16 x float>, align 64
  %16 = alloca <16 x i32>, align 64
  %17 = alloca <16 x i32>, align 64
  %18 = alloca <16 x i32>, align 64
  %19 = alloca <16 x i32>, align 64
  %20 = alloca <16 x i32>, align 64
  %21 = alloca <16 x i32>, align 64
  %22 = alloca <16 x i32>, align 64
  %23 = alloca <16 x i32>, align 64
  %24 = alloca <16 x i32>, align 64
  store <16 x i32> %0, <16 x i32>* %4, align 64, !tbaa !7
  store <16 x i32> %1, <16 x i32>* %5, align 64, !tbaa !7
  store <16 x i32>* %2, <16 x i32>** %6, align 64, !tbaa !7
  %25 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  %26 = uitofp <16 x i32> %25 to <16 x float>
  store <16 x float> %26, <16 x float>* %9, align 64, !tbaa !7
  %27 = load <16 x float>, <16 x float>* %9, align 64, !tbaa !7
  %28 = fptosi <16 x float> %27 to <16 x i32>
  store <16 x i32> %28, <16 x i32>* %17, align 64, !tbaa !7
  %29 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  %30 = load <16 x i32>, <16 x i32>* %17, align 64, !tbaa !7
  %31 = sub <16 x i32> %29, %30
  store <16 x i32> %31, <16 x i32>* %19, align 64, !tbaa !7
  %32 = load <16 x i32>, <16 x i32>* %19, align 64, !tbaa !7
  %33 = uitofp <16 x i32> %32 to <16 x float>
  store <16 x float> %33, <16 x float>* %10, align 64, !tbaa !7
  %34 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %35 = uitofp <16 x i32> %34 to <16 x float>
  store <16 x float> %35, <16 x float>* %7, align 64, !tbaa !7
  %36 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %37 = fptosi <16 x float> %36 to <16 x i32>
  store <16 x i32> %37, <16 x i32>* %16, align 64, !tbaa !7
  %38 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %39 = load <16 x i32>, <16 x i32>* %16, align 64, !tbaa !7
  %40 = sub <16 x i32> %38, %39
  store <16 x i32> %40, <16 x i32>* %18, align 64, !tbaa !7
  %41 = load <16 x i32>, <16 x i32>* %18, align 64, !tbaa !7
  %42 = uitofp <16 x i32> %41 to <16 x float>
  store <16 x float> %42, <16 x float>* %8, align 64, !tbaa !7
  %43 = load <16 x float>, <16 x float>* %9, align 64, !tbaa !7
  %44 = fdiv <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %43
  store <16 x float> %44, <16 x float>* %11, align 64, !tbaa !7
  %45 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %46 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %47 = insertelement <16 x float> undef, float %46, i32 0
  %48 = shufflevector <16 x float> %47, <16 x float> undef, <16 x i32> zeroinitializer
  %49 = fmul <16 x float> %45, %48
  %50 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %51 = fadd <16 x float> %50, %49
  store <16 x float> %51, <16 x float>* %11, align 64, !tbaa !7
  %52 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %53 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %54 = fmul <16 x float> %52, %53
  store <16 x float> %54, <16 x float>* %12, align 64, !tbaa !7
  %55 = load <16 x float>, <16 x float>* %12, align 64, !tbaa !7
  %56 = fptosi <16 x float> %55 to <16 x i32>
  store <16 x i32> %56, <16 x i32>* %20, align 64, !tbaa !7
  %57 = load <16 x i32>, <16 x i32>* %20, align 64, !tbaa !7
  %58 = uitofp <16 x i32> %57 to <16 x float>
  store <16 x float> %58, <16 x float>* %12, align 64, !tbaa !7
  %59 = load <16 x float>, <16 x float>* %7, align 64, !tbaa !7
  %60 = load <16 x float>, <16 x float>* %9, align 64, !tbaa !7
  %61 = load <16 x float>, <16 x float>* %12, align 64, !tbaa !7
  %62 = fmul <16 x float> %60, %61
  %63 = fsub <16 x float> %59, %62
  store <16 x float> %63, <16 x float>* %14, align 64, !tbaa !7
  %64 = load <16 x float>, <16 x float>* %8, align 64, !tbaa !7
  %65 = load <16 x float>, <16 x float>* %10, align 64, !tbaa !7
  %66 = load <16 x float>, <16 x float>* %12, align 64, !tbaa !7
  %67 = fmul <16 x float> %65, %66
  %68 = fsub <16 x float> %64, %67
  store <16 x float> %68, <16 x float>* %15, align 64, !tbaa !7
  %69 = load <16 x float>, <16 x float>* %14, align 64, !tbaa !7
  %70 = load <16 x float>, <16 x float>* %15, align 64, !tbaa !7
  %71 = fadd <16 x float> %69, %70
  store <16 x float> %71, <16 x float>* %15, align 64, !tbaa !7
  %72 = load <16 x float>, <16 x float>* %11, align 64, !tbaa !7
  %73 = load <16 x float>, <16 x float>* %15, align 64, !tbaa !7
  %74 = fmul <16 x float> %72, %73
  store <16 x float> %74, <16 x float>* %13, align 64, !tbaa !7
  %75 = load <16 x float>, <16 x float>* %13, align 64, !tbaa !7
  %76 = fptosi <16 x float> %75 to <16 x i32>
  store <16 x i32> %76, <16 x i32>* %21, align 64, !tbaa !7
  %77 = load <16 x i32>, <16 x i32>* %20, align 64, !tbaa !7
  %78 = load <16 x i32>, <16 x i32>* %21, align 64, !tbaa !7
  %79 = add <16 x i32> %77, %78
  store <16 x i32> %79, <16 x i32>* %22, align 64, !tbaa !7
  %80 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %81 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  %82 = load <16 x i32>, <16 x i32>* %22, align 64, !tbaa !7
  %83 = mul <16 x i32> %81, %82
  %84 = sub <16 x i32> %80, %83
  store <16 x i32> %84, <16 x i32>* %23, align 64, !tbaa !7
  %85 = load <16 x i32>, <16 x i32>* %23, align 64, !tbaa !7
  %86 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  %87 = icmp uge <16 x i32> %85, %86
  %88 = zext <16 x i1> %87 to <16 x i16>
  %89 = trunc <16 x i16> %88 to <16 x i1>
  %90 = select <16 x i1> %89, <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <16 x i32> zeroinitializer
  store <16 x i32> %90, <16 x i32>* %24, align 64, !tbaa !7
  %91 = load <16 x i32>, <16 x i32>* %24, align 64, !tbaa !7
  %92 = and <16 x i32> %91, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %93 = load <16 x i32>, <16 x i32>* %22, align 64, !tbaa !7
  %94 = add <16 x i32> %93, %92
  store <16 x i32> %94, <16 x i32>* %22, align 64, !tbaa !7
  %95 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  %96 = load <16 x i32>, <16 x i32>* %24, align 64, !tbaa !7
  %97 = and <16 x i32> %95, %96
  %98 = load <16 x i32>, <16 x i32>* %23, align 64, !tbaa !7
  %99 = sub <16 x i32> %98, %97
  store <16 x i32> %99, <16 x i32>* %23, align 64, !tbaa !7
  %100 = load <16 x i32>, <16 x i32>* %23, align 64, !tbaa !7
  call void @llvm.genx.vstore.v16i32.p0v16i32(<16 x i32> %100, <16 x i32>* %2)
  %101 = load <16 x i32>, <16 x i32>* %22, align 64, !tbaa !7
  ret <16 x i32> %101
}

; Function Attrs: noinline nounwind
define internal <32 x i32> @_Z24__cm_intrinsic_impl_udivu2CMvb32_jS_(<32 x i32>, <32 x i32>) #19 {
  %3 = alloca <32 x i32>, align 128
  %4 = alloca <32 x i32>, align 128
  %5 = alloca <32 x i32>, align 128
  %6 = alloca <32 x i32>, align 128
  store <32 x i32> %0, <32 x i32>* %3, align 128, !tbaa !7
  store <32 x i32> %1, <32 x i32>* %4, align 128, !tbaa !7
  %7 = load <32 x i32>, <32 x i32>* %3, align 128, !tbaa !7
  %8 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %9 = call <32 x i32> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__jS1_S1_u2CMvrT__j(<32 x i32> %7, <32 x i32> %8, <32 x i32>* %5)
  store <32 x i32> %9, <32 x i32>* %6, align 128, !tbaa !7
  %10 = load <32 x i32>, <32 x i32>* %6, align 128, !tbaa !7
  ret <32 x i32> %10
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <32 x i32> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__jS1_S1_u2CMvrT__j(<32 x i32>, <32 x i32>, <32 x i32>*) #20 comdat {
  %4 = alloca <32 x i32>, align 128
  %5 = alloca <32 x i32>, align 128
  %6 = alloca <32 x i32>*, align 128
  %7 = alloca <32 x float>, align 128
  %8 = alloca <32 x float>, align 128
  %9 = alloca <32 x float>, align 128
  %10 = alloca <32 x float>, align 128
  %11 = alloca <32 x float>, align 128
  %12 = alloca <32 x float>, align 128
  %13 = alloca <32 x float>, align 128
  %14 = alloca <32 x float>, align 128
  %15 = alloca <32 x float>, align 128
  %16 = alloca <32 x i32>, align 128
  %17 = alloca <32 x i32>, align 128
  %18 = alloca <32 x i32>, align 128
  %19 = alloca <32 x i32>, align 128
  %20 = alloca <32 x i32>, align 128
  %21 = alloca <32 x i32>, align 128
  %22 = alloca <32 x i32>, align 128
  %23 = alloca <32 x i32>, align 128
  %24 = alloca <32 x i32>, align 128
  store <32 x i32> %0, <32 x i32>* %4, align 128, !tbaa !7
  store <32 x i32> %1, <32 x i32>* %5, align 128, !tbaa !7
  store <32 x i32>* %2, <32 x i32>** %6, align 128, !tbaa !7
  %25 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  %26 = uitofp <32 x i32> %25 to <32 x float>
  store <32 x float> %26, <32 x float>* %9, align 128, !tbaa !7
  %27 = load <32 x float>, <32 x float>* %9, align 128, !tbaa !7
  %28 = fptosi <32 x float> %27 to <32 x i32>
  store <32 x i32> %28, <32 x i32>* %17, align 128, !tbaa !7
  %29 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  %30 = load <32 x i32>, <32 x i32>* %17, align 128, !tbaa !7
  %31 = sub <32 x i32> %29, %30
  store <32 x i32> %31, <32 x i32>* %19, align 128, !tbaa !7
  %32 = load <32 x i32>, <32 x i32>* %19, align 128, !tbaa !7
  %33 = uitofp <32 x i32> %32 to <32 x float>
  store <32 x float> %33, <32 x float>* %10, align 128, !tbaa !7
  %34 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %35 = uitofp <32 x i32> %34 to <32 x float>
  store <32 x float> %35, <32 x float>* %7, align 128, !tbaa !7
  %36 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %37 = fptosi <32 x float> %36 to <32 x i32>
  store <32 x i32> %37, <32 x i32>* %16, align 128, !tbaa !7
  %38 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %39 = load <32 x i32>, <32 x i32>* %16, align 128, !tbaa !7
  %40 = sub <32 x i32> %38, %39
  store <32 x i32> %40, <32 x i32>* %18, align 128, !tbaa !7
  %41 = load <32 x i32>, <32 x i32>* %18, align 128, !tbaa !7
  %42 = uitofp <32 x i32> %41 to <32 x float>
  store <32 x float> %42, <32 x float>* %8, align 128, !tbaa !7
  %43 = load <32 x float>, <32 x float>* %9, align 128, !tbaa !7
  %44 = fdiv <32 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %43
  store <32 x float> %44, <32 x float>* %11, align 128, !tbaa !7
  %45 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %46 = call float @_ZN7details16__impl_hex2floatEj(i32 -1262485504)
  %47 = insertelement <32 x float> undef, float %46, i32 0
  %48 = shufflevector <32 x float> %47, <32 x float> undef, <32 x i32> zeroinitializer
  %49 = fmul <32 x float> %45, %48
  %50 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %51 = fadd <32 x float> %50, %49
  store <32 x float> %51, <32 x float>* %11, align 128, !tbaa !7
  %52 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %53 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %54 = fmul <32 x float> %52, %53
  store <32 x float> %54, <32 x float>* %12, align 128, !tbaa !7
  %55 = load <32 x float>, <32 x float>* %12, align 128, !tbaa !7
  %56 = fptosi <32 x float> %55 to <32 x i32>
  store <32 x i32> %56, <32 x i32>* %20, align 128, !tbaa !7
  %57 = load <32 x i32>, <32 x i32>* %20, align 128, !tbaa !7
  %58 = uitofp <32 x i32> %57 to <32 x float>
  store <32 x float> %58, <32 x float>* %12, align 128, !tbaa !7
  %59 = load <32 x float>, <32 x float>* %7, align 128, !tbaa !7
  %60 = load <32 x float>, <32 x float>* %9, align 128, !tbaa !7
  %61 = load <32 x float>, <32 x float>* %12, align 128, !tbaa !7
  %62 = fmul <32 x float> %60, %61
  %63 = fsub <32 x float> %59, %62
  store <32 x float> %63, <32 x float>* %14, align 128, !tbaa !7
  %64 = load <32 x float>, <32 x float>* %8, align 128, !tbaa !7
  %65 = load <32 x float>, <32 x float>* %10, align 128, !tbaa !7
  %66 = load <32 x float>, <32 x float>* %12, align 128, !tbaa !7
  %67 = fmul <32 x float> %65, %66
  %68 = fsub <32 x float> %64, %67
  store <32 x float> %68, <32 x float>* %15, align 128, !tbaa !7
  %69 = load <32 x float>, <32 x float>* %14, align 128, !tbaa !7
  %70 = load <32 x float>, <32 x float>* %15, align 128, !tbaa !7
  %71 = fadd <32 x float> %69, %70
  store <32 x float> %71, <32 x float>* %15, align 128, !tbaa !7
  %72 = load <32 x float>, <32 x float>* %11, align 128, !tbaa !7
  %73 = load <32 x float>, <32 x float>* %15, align 128, !tbaa !7
  %74 = fmul <32 x float> %72, %73
  store <32 x float> %74, <32 x float>* %13, align 128, !tbaa !7
  %75 = load <32 x float>, <32 x float>* %13, align 128, !tbaa !7
  %76 = fptosi <32 x float> %75 to <32 x i32>
  store <32 x i32> %76, <32 x i32>* %21, align 128, !tbaa !7
  %77 = load <32 x i32>, <32 x i32>* %20, align 128, !tbaa !7
  %78 = load <32 x i32>, <32 x i32>* %21, align 128, !tbaa !7
  %79 = add <32 x i32> %77, %78
  store <32 x i32> %79, <32 x i32>* %22, align 128, !tbaa !7
  %80 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %81 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  %82 = load <32 x i32>, <32 x i32>* %22, align 128, !tbaa !7
  %83 = mul <32 x i32> %81, %82
  %84 = sub <32 x i32> %80, %83
  store <32 x i32> %84, <32 x i32>* %23, align 128, !tbaa !7
  %85 = load <32 x i32>, <32 x i32>* %23, align 128, !tbaa !7
  %86 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  %87 = icmp uge <32 x i32> %85, %86
  %88 = zext <32 x i1> %87 to <32 x i16>
  %89 = trunc <32 x i16> %88 to <32 x i1>
  %90 = select <32 x i1> %89, <32 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <32 x i32> zeroinitializer
  store <32 x i32> %90, <32 x i32>* %24, align 128, !tbaa !7
  %91 = load <32 x i32>, <32 x i32>* %24, align 128, !tbaa !7
  %92 = and <32 x i32> %91, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %93 = load <32 x i32>, <32 x i32>* %22, align 128, !tbaa !7
  %94 = add <32 x i32> %93, %92
  store <32 x i32> %94, <32 x i32>* %22, align 128, !tbaa !7
  %95 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  %96 = load <32 x i32>, <32 x i32>* %24, align 128, !tbaa !7
  %97 = and <32 x i32> %95, %96
  %98 = load <32 x i32>, <32 x i32>* %23, align 128, !tbaa !7
  %99 = sub <32 x i32> %98, %97
  store <32 x i32> %99, <32 x i32>* %23, align 128, !tbaa !7
  %100 = load <32 x i32>, <32 x i32>* %23, align 128, !tbaa !7
  call void @llvm.genx.vstore.v32i32.p0v32i32(<32 x i32> %100, <32 x i32>* %2)
  %101 = load <32 x i32>, <32 x i32>* %22, align 128, !tbaa !7
  ret <32 x i32> %101
}

; Function Attrs: noinline nounwind
define internal i32 @_Z24__cm_intrinsic_impl_uremjj(i32, i32) #14 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca <1 x i32>, align 4
  %6 = alloca <1 x i32>, align 4
  %7 = alloca <1 x i32>, align 4
  store i32 %0, i32* %3, align 4, !tbaa !13
  store i32 %1, i32* %4, align 4, !tbaa !13
  %8 = load i32, i32* %3, align 4, !tbaa !13
  %9 = insertelement <1 x i32> undef, i32 %8, i32 0
  %10 = shufflevector <1 x i32> %9, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %10, <1 x i32>* %5, align 4, !tbaa !7
  %11 = load i32, i32* %4, align 4, !tbaa !13
  %12 = insertelement <1 x i32> undef, i32 %11, i32 0
  %13 = shufflevector <1 x i32> %12, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %13, <1 x i32>* %6, align 4, !tbaa !7
  %14 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  %15 = load <1 x i32>, <1 x i32>* %6, align 4, !tbaa !7
  %16 = call <1 x i32> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__jS1_S1_u2CMvrT__j(<1 x i32> %14, <1 x i32> %15, <1 x i32>* %7)
  %17 = load <1 x i32>, <1 x i32>* %7, align 4, !tbaa !7
  %18 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v1i32.i16(<1 x i32> %17, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %19 = extractelement <1 x i32> %18, i32 0
  ret i32 %19
}

; Function Attrs: noinline nounwind
define internal <1 x i32> @_Z24__cm_intrinsic_impl_uremu2CMvb1_jS_(<1 x i32>, <1 x i32>) #14 {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x i32>, align 4
  %5 = alloca <1 x i32>, align 4
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %4, align 4, !tbaa !7
  %6 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %7 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  %8 = call <1 x i32> @_ZN7details14__impl_udivremILi1EEEu2CMvbT__jS1_S1_u2CMvrT__j(<1 x i32> %6, <1 x i32> %7, <1 x i32>* %5)
  %9 = load <1 x i32>, <1 x i32>* %5, align 4, !tbaa !7
  ret <1 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <2 x i32> @_Z24__cm_intrinsic_impl_uremu2CMvb2_jS_(<2 x i32>, <2 x i32>) #15 {
  %3 = alloca <2 x i32>, align 8
  %4 = alloca <2 x i32>, align 8
  %5 = alloca <2 x i32>, align 8
  store <2 x i32> %0, <2 x i32>* %3, align 8, !tbaa !7
  store <2 x i32> %1, <2 x i32>* %4, align 8, !tbaa !7
  %6 = load <2 x i32>, <2 x i32>* %3, align 8, !tbaa !7
  %7 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  %8 = call <2 x i32> @_ZN7details14__impl_udivremILi2EEEu2CMvbT__jS1_S1_u2CMvrT__j(<2 x i32> %6, <2 x i32> %7, <2 x i32>* %5)
  %9 = load <2 x i32>, <2 x i32>* %5, align 8, !tbaa !7
  ret <2 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <4 x i32> @_Z24__cm_intrinsic_impl_uremu2CMvb4_jS_(<4 x i32>, <4 x i32>) #16 {
  %3 = alloca <4 x i32>, align 16
  %4 = alloca <4 x i32>, align 16
  %5 = alloca <4 x i32>, align 16
  store <4 x i32> %0, <4 x i32>* %3, align 16, !tbaa !7
  store <4 x i32> %1, <4 x i32>* %4, align 16, !tbaa !7
  %6 = load <4 x i32>, <4 x i32>* %3, align 16, !tbaa !7
  %7 = load <4 x i32>, <4 x i32>* %4, align 16, !tbaa !7
  %8 = call <4 x i32> @_ZN7details14__impl_udivremILi4EEEu2CMvbT__jS1_S1_u2CMvrT__j(<4 x i32> %6, <4 x i32> %7, <4 x i32>* %5)
  %9 = load <4 x i32>, <4 x i32>* %5, align 16, !tbaa !7
  ret <4 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <8 x i32> @_Z24__cm_intrinsic_impl_uremu2CMvb8_jS_(<8 x i32>, <8 x i32>) #17 {
  %3 = alloca <8 x i32>, align 32
  %4 = alloca <8 x i32>, align 32
  %5 = alloca <8 x i32>, align 32
  store <8 x i32> %0, <8 x i32>* %3, align 32, !tbaa !7
  store <8 x i32> %1, <8 x i32>* %4, align 32, !tbaa !7
  %6 = load <8 x i32>, <8 x i32>* %3, align 32, !tbaa !7
  %7 = load <8 x i32>, <8 x i32>* %4, align 32, !tbaa !7
  %8 = call <8 x i32> @_ZN7details14__impl_udivremILi8EEEu2CMvbT__jS1_S1_u2CMvrT__j(<8 x i32> %6, <8 x i32> %7, <8 x i32>* %5)
  %9 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  ret <8 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <16 x i32> @_Z24__cm_intrinsic_impl_uremu2CMvb16_jS_(<16 x i32>, <16 x i32>) #18 {
  %3 = alloca <16 x i32>, align 64
  %4 = alloca <16 x i32>, align 64
  %5 = alloca <16 x i32>, align 64
  store <16 x i32> %0, <16 x i32>* %3, align 64, !tbaa !7
  store <16 x i32> %1, <16 x i32>* %4, align 64, !tbaa !7
  %6 = load <16 x i32>, <16 x i32>* %3, align 64, !tbaa !7
  %7 = load <16 x i32>, <16 x i32>* %4, align 64, !tbaa !7
  %8 = call <16 x i32> @_ZN7details14__impl_udivremILi16EEEu2CMvbT__jS1_S1_u2CMvrT__j(<16 x i32> %6, <16 x i32> %7, <16 x i32>* %5)
  %9 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !7
  ret <16 x i32> %9
}

; Function Attrs: noinline nounwind
define internal <32 x i32> @_Z24__cm_intrinsic_impl_uremu2CMvb32_jS_(<32 x i32>, <32 x i32>) #19 {
  %3 = alloca <32 x i32>, align 128
  %4 = alloca <32 x i32>, align 128
  %5 = alloca <32 x i32>, align 128
  store <32 x i32> %0, <32 x i32>* %3, align 128, !tbaa !7
  store <32 x i32> %1, <32 x i32>* %4, align 128, !tbaa !7
  %6 = load <32 x i32>, <32 x i32>* %3, align 128, !tbaa !7
  %7 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !7
  %8 = call <32 x i32> @_ZN7details14__impl_udivremILi32EEEu2CMvbT__jS1_S1_u2CMvrT__j(<32 x i32> %6, <32 x i32> %7, <32 x i32>* %5)
  %9 = load <32 x i32>, <32 x i32>* %5, align 128, !tbaa !7
  ret <32 x i32> %9
}

; Function Attrs: noinline nounwind
define internal void @__do_print_cm(i8*, i8*, i32, i64, i32*, i32, i32) #23 !dbg !15 {
  %8 = alloca i8*, align 4
  %9 = alloca i8*, align 4
  %10 = alloca i32, align 4
  %11 = alloca i64, align 8
  %12 = alloca i32*, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca %class.ArgWriter, align 8
  %20 = alloca <128 x i8>, align 128
  store i8* %0, i8** %8, align 4, !tbaa !17
  store i8* %1, i8** %9, align 4, !tbaa !17
  store i32 %2, i32* %10, align 4, !tbaa !13
  store i64 %3, i64* %11, align 8, !tbaa !19
  store i32* %4, i32** %12, align 4, !tbaa !17
  store i32 %5, i32* %13, align 4, !tbaa !13
  store i32 %6, i32* %14, align 4, !tbaa !13
  %21 = load i32, i32* %13, align 4, !dbg !21, !tbaa !13
  %22 = load i32, i32* %14, align 4, !dbg !22, !tbaa !13
  %23 = load i32, i32* %10, align 4, !dbg !23, !tbaa !13
  %24 = mul nsw i32 %22, %23, !dbg !24
  %25 = add nsw i32 %21, %24, !dbg !25
  %26 = mul nsw i32 %25, 32, !dbg !26
  %27 = add nsw i32 160, %26, !dbg !27
  store i32 %27, i32* %15, align 4, !dbg !28, !tbaa !13
  %28 = call i32 @llvm.genx.predefined.surface(i32 2), !dbg !29
  store i32 %28, i32* %16, align 4, !dbg !30, !tbaa !31
  %29 = load i32, i32* %16, align 4, !dbg !33, !tbaa !31
  %30 = load i32, i32* %15, align 4, !dbg !34, !tbaa !13
  %31 = call i32 @_ZN7details21_cm_print_init_offsetE15cm_surfaceindexj(i32 %29, i32 %30), !dbg !35
  store i32 %31, i32* %17, align 4, !dbg !36, !tbaa !13
  %32 = load i32, i32* %17, align 4, !dbg !37, !tbaa !13
  %33 = add i32 %32, 128, !dbg !38
  %34 = add i32 %33, 32, !dbg !39
  store i32 %34, i32* %18, align 4, !dbg !40, !tbaa !13
  %35 = load i32, i32* %18, align 4, !dbg !41, !tbaa !13
  %36 = load i32*, i32** %12, align 4, !dbg !42, !tbaa !17
  %37 = load i32, i32* %10, align 4, !dbg !43, !tbaa !13
  %38 = load i64, i64* %11, align 8, !dbg !44, !tbaa !19
  call void @_ZN9ArgWriterC2EjPKjiy(%class.ArgWriter* %19, i32 %35, i32* %36, i32 %37, i64 %38), !dbg !45
  %39 = load i8*, i8** %8, align 4, !dbg !46, !tbaa !17
  %40 = load i8*, i8** %9, align 4, !dbg !47, !tbaa !17
  %41 = call <128 x i8> @_Z14GetFormatedStrI9ArgWriterEu2CMvb128_cPKcS3_RT_(i8* %39, i8* %40, %class.ArgWriter* dereferenceable(24) %19), !dbg !48
  store <128 x i8> %41, <128 x i8>* %20, align 128, !dbg !49, !tbaa !7
  %42 = load i32, i32* %16, align 4, !dbg !50, !tbaa !31
  %43 = load i32, i32* %17, align 4, !dbg !51, !tbaa !13
  %44 = load <128 x i8>, <128 x i8>* %20, align 128, !dbg !52, !tbaa !7
  call void @_ZN7details16_cm_print_formatILi128EEEv15cm_surfaceindexju2CMvbT__c(i32 %42, i32 %43, <128 x i8> %44), !dbg !53
  ret void, !dbg !54
}

declare dso_local i32 @_ZN7details38__cm_intrinsic_impl_predefined_surfaceEj(i32) #24

; Function Attrs: nounwind readnone
declare i32 @llvm.genx.predefined.surface(i32) #2

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details21_cm_print_init_offsetE15cm_surfaceindexj(i32, i32) #10 comdat {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca <8 x i32>, align 32
  %6 = alloca <8 x i32>, align 32
  %7 = alloca <8 x i32>, align 32
  store i32 %0, i32* %3, align 4, !tbaa !31
  store i32 %1, i32* %4, align 4, !tbaa !13
  store <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>, <8 x i32>* %5, align 32, !tbaa !7
  store <8 x i32> zeroinitializer, <8 x i32>* %6, align 32, !tbaa !7
  %8 = load i32, i32* %4, align 4, !tbaa !13
  %9 = load <8 x i32>, <8 x i32>* %6, align 32, !tbaa !7
  %10 = insertelement <1 x i32> undef, i32 %8, i32 0
  %11 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %9, <1 x i32> %10, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %11, <8 x i32>* %6, align 32, !tbaa !7
  store <8 x i32> zeroinitializer, <8 x i32>* %7, align 32, !tbaa !7
  %12 = load i32, i32* %3, align 4, !tbaa !31
  %13 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %14 = load <8 x i32>, <8 x i32>* %6, align 32, !tbaa !7
  call void @_Z12write_atomicIL14CmAtomicOpType0EjLi8EENSt9enable_ifIXaaaaaaaaaaneT_LS0_7EneT_LS0_18EneT_LS0_2EneT_LS0_3EneT_LS0_255EclL_ZN7detailsL10isPowerOf2EjjET1_Li32EEEvE4typeE15cm_surfaceindexu2CMvbT1__ju2CMvbT1__T0_u2CMvrT1__S6_(i32 %12, <8 x i32> %13, <8 x i32> %14, <8 x i32>* %7)
  %15 = load <8 x i32>, <8 x i32>* %7, align 32, !tbaa !7
  %16 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v8i32.i16(<8 x i32> %15, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %17 = extractelement <1 x i32> %16, i32 0
  ret i32 %17
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriterC2EjPKjiy(%class.ArgWriter*, i32, i32*, i32, i64) unnamed_addr #25 comdat align 2 !dbg !56 {
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32*, align 4
  %9 = alloca i32, align 4
  %10 = alloca i64, align 8
  store %class.ArgWriter* %0, %class.ArgWriter** %6, align 4, !tbaa !17
  store i32 %1, i32* %7, align 4, !tbaa !13
  store i32* %2, i32** %8, align 4, !tbaa !17
  store i32 %3, i32* %9, align 4, !tbaa !13
  store i64 %4, i64* %10, align 8, !tbaa !19
  %11 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4
  %12 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %11, i32 0, i32 0, !dbg !57
  %13 = load i32, i32* %7, align 4, !dbg !58, !tbaa !13
  store i32 %13, i32* %12, align 8, !dbg !57, !tbaa !59
  %14 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %11, i32 0, i32 1, !dbg !61
  %15 = load i32*, i32** %8, align 4, !dbg !62, !tbaa !17
  store i32* %15, i32** %14, align 4, !dbg !61, !tbaa !63
  %16 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %11, i32 0, i32 2, !dbg !64
  store i32 0, i32* %16, align 8, !dbg !64, !tbaa !65
  %17 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %11, i32 0, i32 3, !dbg !66
  %18 = load i32, i32* %9, align 4, !dbg !67, !tbaa !13
  store i32 %18, i32* %17, align 4, !dbg !66, !tbaa !68
  %19 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %11, i32 0, i32 4, !dbg !69
  %20 = load i64, i64* %10, align 8, !dbg !70, !tbaa !19
  store i64 %20, i64* %19, align 8, !dbg !69, !tbaa !71
  ret void, !dbg !72
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <128 x i8> @_Z14GetFormatedStrI9ArgWriterEu2CMvb128_cPKcS3_RT_(i8*, i8*, %class.ArgWriter* dereferenceable(24)) #20 comdat !dbg !73 {
  %4 = alloca i8*, align 4
  %5 = alloca i8*, align 4
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <128 x i8>, align 128
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca <100 x i8>, align 128
  store i8* %0, i8** %4, align 4, !tbaa !17
  store i8* %1, i8** %5, align 4, !tbaa !17
  store %class.ArgWriter* %2, %class.ArgWriter** %6, align 4, !tbaa !17
  store i32 0, i32* %8, align 4, !dbg !75, !tbaa !13
  store i32 127, i32* %9, align 4, !dbg !76, !tbaa !13
  br label %12, !dbg !77

; <label>:12:                                     ; preds = %48, %3
  %13 = load i8*, i8** %4, align 4, !dbg !78, !tbaa !17
  %14 = load i32, i32* %8, align 4, !dbg !79, !tbaa !13
  %15 = load i32, i32* %9, align 4, !dbg !80, !tbaa !13
  %16 = call i32 @_Z13CopyPlainTextILj128EEiPKcu2CMvrT__cii(i8* %13, <128 x i8>* %7, i32 %14, i32 %15), !dbg !81
  store i32 %16, i32* %10, align 4, !dbg !82, !tbaa !13
  %17 = load i32, i32* %10, align 4, !dbg !83, !tbaa !13
  %18 = load i8*, i8** %4, align 4, !dbg !84, !tbaa !17
  %19 = getelementptr inbounds i8, i8* %18, i32 %17, !dbg !84
  store i8* %19, i8** %4, align 4, !dbg !84, !tbaa !17
  %20 = load i32, i32* %10, align 4, !dbg !85, !tbaa !13
  %21 = load i32, i32* %8, align 4, !dbg !86, !tbaa !13
  %22 = add nsw i32 %21, %20, !dbg !86
  store i32 %22, i32* %8, align 4, !dbg !86, !tbaa !13
  %23 = load i32, i32* %10, align 4, !dbg !87, !tbaa !13
  %24 = load i32, i32* %9, align 4, !dbg !88, !tbaa !13
  %25 = sub nsw i32 %24, %23, !dbg !88
  store i32 %25, i32* %9, align 4, !dbg !88, !tbaa !13
  %26 = load i32, i32* %9, align 4, !dbg !89, !tbaa !13
  %27 = icmp ne i32 %26, 0, !dbg !89
  br i1 %27, label %28, label %33, !dbg !90

; <label>:28:                                     ; preds = %12
  %29 = load i8*, i8** %4, align 4, !dbg !91, !tbaa !17
  %30 = load i8, i8* %29, align 1, !dbg !92, !tbaa !7
  %31 = sext i8 %30 to i32, !dbg !92
  %32 = icmp eq i32 %31, 0, !dbg !93
  br i1 %32, label %33, label %34, !dbg !94

; <label>:33:                                     ; preds = %28, %12
  br label %53, !dbg !95

; <label>:34:                                     ; preds = %28
  %35 = load i8*, i8** %5, align 4, !dbg !96, !tbaa !17
  %36 = load i8, i8* %35, align 1, !dbg !97, !tbaa !7
  %37 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !98, !tbaa !17
  %38 = call <100 x i8> @_ZN7details7Arg2StrI9ArgWriterEEu2CMvb100_ccRT_(i8 signext %36, %class.ArgWriter* dereferenceable(24) %37), !dbg !99
  store <100 x i8> %38, <100 x i8>* %11, align 128, !dbg !100, !tbaa !7
  %39 = load i32, i32* %8, align 4, !dbg !101, !tbaa !13
  %40 = load i32, i32* %9, align 4, !dbg !102, !tbaa !13
  %41 = call i32 @_Z12CopyFullTextILj100ELj128EEiu2CMvrT__ciu2CMvrT0__cii(<100 x i8>* %11, i32 0, <128 x i8>* %7, i32 %39, i32 %40), !dbg !103
  store i32 %41, i32* %10, align 4, !dbg !104, !tbaa !13
  %42 = load i32, i32* %10, align 4, !dbg !105, !tbaa !13
  %43 = load i32, i32* %8, align 4, !dbg !106, !tbaa !13
  %44 = add nsw i32 %43, %42, !dbg !106
  store i32 %44, i32* %8, align 4, !dbg !106, !tbaa !13
  %45 = load i32, i32* %10, align 4, !dbg !107, !tbaa !13
  %46 = load i32, i32* %9, align 4, !dbg !108, !tbaa !13
  %47 = sub nsw i32 %46, %45, !dbg !108
  store i32 %47, i32* %9, align 4, !dbg !108, !tbaa !13
  br label %48, !dbg !109

; <label>:48:                                     ; preds = %34
  %49 = load i8*, i8** %4, align 4, !dbg !110, !tbaa !17
  %50 = getelementptr inbounds i8, i8* %49, i32 1, !dbg !110
  store i8* %50, i8** %4, align 4, !dbg !110, !tbaa !17
  %51 = load i8*, i8** %5, align 4, !dbg !111, !tbaa !17
  %52 = getelementptr inbounds i8, i8* %51, i32 1, !dbg !111
  store i8* %52, i8** %5, align 4, !dbg !111, !tbaa !17
  br label %12, !dbg !77, !llvm.loop !112

; <label>:53:                                     ; preds = %33
  %54 = load i32, i32* %8, align 4, !dbg !113, !tbaa !13
  %55 = trunc i32 %54 to i16, !dbg !113
  %56 = load <128 x i8>, <128 x i8>* %7, align 128, !dbg !114, !tbaa !7
  %57 = mul i16 %55, 1, !dbg !114
  %58 = call <128 x i8> @llvm.genx.wrregioni.v128i8.v1i8.i16.i1(<128 x i8> %56, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %57, i32 undef, i1 true), !dbg !114
  store <128 x i8> %58, <128 x i8>* %7, align 128, !dbg !114, !tbaa !7
  %59 = load <128 x i8>, <128 x i8>* %7, align 128, !dbg !115, !tbaa !7
  ret <128 x i8> %59, !dbg !116
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details16_cm_print_formatILi128EEEv15cm_surfaceindexju2CMvbT__c(i32, i32, <128 x i8>) #20 comdat {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca <128 x i8>, align 128
  %7 = alloca <8 x i32>, align 32
  store i32 %0, i32* %4, align 4, !tbaa !31
  store i32 %1, i32* %5, align 4, !tbaa !13
  store <128 x i8> %2, <128 x i8>* %6, align 128, !tbaa !7
  store <8 x i32> zeroinitializer, <8 x i32>* %7, align 32, !tbaa !7
  %8 = load <8 x i32>, <8 x i32>* %7, align 32, !tbaa !7
  %9 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %8, <1 x i32> <i32 5>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %9, <8 x i32>* %7, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %7, align 32, !tbaa !7
  %11 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %10, <1 x i32> zeroinitializer, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true)
  store <8 x i32> %11, <8 x i32>* %7, align 32, !tbaa !7
  %12 = load <8 x i32>, <8 x i32>* %7, align 32, !tbaa !7
  %13 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %12, <1 x i32> <i32 128>, i32 0, i32 1, i32 0, i16 24, i32 undef, i1 true)
  store <8 x i32> %13, <8 x i32>* %7, align 32, !tbaa !7
  %14 = load i32, i32* %4, align 4, !tbaa !31
  %15 = load i32, i32* %5, align 4, !tbaa !13
  %16 = load <8 x i32>, <8 x i32>* %7, align 32, !tbaa !7
  call void @_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_(i32 %14, i32 %15, <8 x i32> %16)
  %17 = load i32, i32* %5, align 4, !tbaa !13
  %18 = add i32 %17, 32
  store i32 %18, i32* %5, align 4, !tbaa !13
  %19 = load i32, i32* %4, align 4, !tbaa !31
  %20 = load i32, i32* %5, align 4, !tbaa !13
  %21 = load <128 x i8>, <128 x i8>* %6, align 128, !tbaa !7
  call void @_Z5writeIcLi128EEv15cm_surfaceindexiu2CMvbT0__T_(i32 %19, i32 %20, <128 x i8> %21)
  ret void
}

; Function Attrs: noinline nounwind
define internal void @__do_print_lz(i32, i8*, i32, i64, i32*, i32, i32, i32, i32) #26 !dbg !117 {
  %10 = alloca i32, align 4
  %11 = alloca i8*, align 4
  %12 = alloca i32, align 4
  %13 = alloca i64, align 8
  %14 = alloca i32*, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = alloca <5 x i32>, align 32
  %22 = alloca %class.UniformWriter, align 32
  %23 = alloca %class.VaryingWriter, align 32
  store i32 %0, i32* %10, align 4, !tbaa !13
  store i8* %1, i8** %11, align 4, !tbaa !17
  store i32 %2, i32* %12, align 4, !tbaa !13
  store i64 %3, i64* %13, align 8, !tbaa !19
  store i32* %4, i32** %14, align 4, !tbaa !17
  store i32 %5, i32* %15, align 4, !tbaa !13
  store i32 %6, i32* %16, align 4, !tbaa !13
  store i32 %7, i32* %17, align 4, !tbaa !13
  store i32 %8, i32* %18, align 4, !tbaa !13
  %24 = call i64 @llvm.genx.print.buffer(), !dbg !118
  %25 = trunc i64 %24 to i32, !dbg !118
  store i32 %25, i32* %19, align 4, !dbg !119, !tbaa !13
  %26 = load i32, i32* %15, align 4, !dbg !120, !tbaa !13
  %27 = load i32, i32* %12, align 4, !dbg !121, !tbaa !13
  %28 = mul nsw i32 3, %27, !dbg !122
  %29 = load i32, i32* %16, align 4, !dbg !123, !tbaa !13
  %30 = mul nsw i32 %28, %29, !dbg !124
  %31 = add nsw i32 %26, %30, !dbg !125
  %32 = mul nsw i32 %31, 8, !dbg !126
  %33 = add nsw i32 4, %32, !dbg !127
  %34 = load i32, i32* %17, align 4, !dbg !128, !tbaa !13
  %35 = load i32, i32* %12, align 4, !dbg !129, !tbaa !13
  %36 = load i32, i32* %18, align 4, !dbg !130, !tbaa !13
  %37 = mul nsw i32 %35, %36, !dbg !131
  %38 = add nsw i32 %34, %37, !dbg !132
  %39 = mul nsw i32 %38, 4, !dbg !133
  %40 = add nsw i32 %33, %39, !dbg !134
  store i32 %40, i32* %20, align 4, !dbg !135, !tbaa !13
  %41 = load i32, i32* %19, align 4, !dbg !136, !tbaa !13
  %42 = insertelement <1 x i32> undef, i32 %41, i32 0, !dbg !136
  %43 = shufflevector <1 x i32> %42, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !136
  %44 = load i32, i32* %20, align 4, !dbg !137, !tbaa !13
  %45 = call i32 @_ZN7details25_cm_print_init_offset_oclEu2CMvb1_jj(<1 x i32> %43, i32 %44), !dbg !138
  %46 = load i32, i32* %19, align 4, !dbg !139, !tbaa !13
  %47 = add i32 %46, %45, !dbg !139
  store i32 %47, i32* %19, align 4, !dbg !139, !tbaa !13
  %48 = load i32, i32* %19, align 4, !dbg !140, !tbaa !13
  %49 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %48), !dbg !141
  %50 = load i32, i32* %10, align 4, !dbg !142, !tbaa !13
  %51 = call <1 x i32> @_ZL11vector_castIiEu2CMvb1_T_S0_(i32 %50), !dbg !143
  call void @_Z20cm_svm_scatter_writeIiLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %49, <1 x i32> %51), !dbg !144
  %52 = load i32, i32* %19, align 4, !dbg !145, !tbaa !13
  %53 = add i32 %52, 4, !dbg !145
  store i32 %53, i32* %19, align 4, !dbg !145, !tbaa !13
  %54 = call <5 x i32> @_Z17get_auxiliary_strv(), !dbg !146
  store <5 x i32> %54, <5 x i32>* %21, align 32, !dbg !147, !tbaa !7
  br label %55, !dbg !148

; <label>:55:                                     ; preds = %70, %9
  %56 = load i8*, i8** %11, align 4, !dbg !149, !tbaa !17
  %57 = load i8, i8* %56, align 1, !dbg !150, !tbaa !7
  %58 = sext i8 %57 to i32, !dbg !150
  %59 = icmp ne i32 %58, 0, !dbg !151
  br i1 %59, label %60, label %73, !dbg !148

; <label>:60:                                     ; preds = %55
  %61 = load i8*, i8** %11, align 4, !dbg !152, !tbaa !17
  %62 = load i8, i8* %61, align 1, !dbg !153, !tbaa !7
  %63 = load i32, i32* %12, align 4, !dbg !154, !tbaa !13
  %64 = load i64, i64* %13, align 8, !dbg !155, !tbaa !19
  %65 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !156, !tbaa !7
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %22, i32* dereferenceable(4) %19, i32** dereferenceable(4) %14, i32 %63, i64 %64, <5 x i32> %65), !dbg !157
  %66 = load i32, i32* %12, align 4, !dbg !158, !tbaa !13
  %67 = load i64, i64* %13, align 8, !dbg !159, !tbaa !19
  %68 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !160, !tbaa !7
  call void @_ZN13VaryingWriterC2ERjRPKjiyu2CMvb5_i(%class.VaryingWriter* %23, i32* dereferenceable(4) %19, i32** dereferenceable(4) %14, i32 %66, i64 %67, <5 x i32> %68), !dbg !161
  %69 = call zeroext i1 @_ZN9PrintInfo14switchEncodingI13UniformWriter13VaryingWriterEEbNS_8EncodingET_T0_(i8 signext %62, %class.UniformWriter* byval(%class.UniformWriter) align 32 %22, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %23), !dbg !162
  br label %70, !dbg !163

; <label>:70:                                     ; preds = %60
  %71 = load i8*, i8** %11, align 4, !dbg !164, !tbaa !17
  %72 = getelementptr inbounds i8, i8* %71, i32 1, !dbg !164
  store i8* %72, i8** %11, align 4, !dbg !164, !tbaa !17
  br label %55, !dbg !148, !llvm.loop !165

; <label>:73:                                     ; preds = %55
  ret void, !dbg !166
}

; Function Attrs: nounwind
declare i64 @llvm.genx.print.buffer() #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details25_cm_print_init_offset_oclEu2CMvb1_jj(<1 x i32>, i32) #10 comdat {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca i32, align 4
  %5 = alloca <8 x i32>, align 32
  %6 = alloca i32, align 4
  %7 = alloca <8 x i32>, align 32
  %8 = alloca <8 x i16>, align 16
  %9 = alloca <8 x i32>, align 32
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store i32 %1, i32* %4, align 4, !tbaa !13
  store i32 0, i32* %6, align 4, !tbaa !168
  store <8 x i32> zeroinitializer, <8 x i32>* %7, align 32, !tbaa !7
  store <8 x i16> <i16 0, i16 4, i16 8, i16 12, i16 16, i16 20, i16 24, i16 28>, <8 x i16>* %8, align 16, !tbaa !7
  %10 = load i32, i32* %4, align 4, !tbaa !13
  %11 = load <8 x i32>, <8 x i32>* %7, align 32, !tbaa !7
  %12 = insertelement <1 x i32> undef, i32 %10, i32 0
  %13 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %11, <1 x i32> %12, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %13, <8 x i32>* %7, align 32, !tbaa !7
  %14 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %15 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v1i32.i16(<1 x i32> %14, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %16 = extractelement <1 x i32> %15, i32 0
  %17 = insertelement <8 x i32> undef, i32 %16, i32 0
  %18 = shufflevector <8 x i32> %17, <8 x i32> undef, <8 x i32> zeroinitializer
  %19 = load <8 x i16>, <8 x i16>* %8, align 16, !tbaa !7
  %20 = zext <8 x i16> %19 to <8 x i32>
  %21 = add <8 x i32> %18, %20
  store <8 x i32> %21, <8 x i32>* %9, align 32, !tbaa !7
  %22 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %23 = load <8 x i32>, <8 x i32>* %7, align 32, !tbaa !7
  %24 = load <8 x i32>, <8 x i32>* %5
  %25 = call <8 x i32> @llvm.genx.svm.atomic.add.v8i32.v8i1.v8i32(<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> %22, <8 x i32> %23, <8 x i32> %24)
  store <8 x i32> %25, <8 x i32>* %5
  %26 = load <8 x i32>, <8 x i32>* %5, align 32, !tbaa !7
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v8i32.i16(<8 x i32> %26, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %28 = extractelement <1 x i32> %27, i32 0
  ret i32 %28
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_Z20cm_svm_scatter_writeIiLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32>, <1 x i32>) #4 comdat {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x i32>, align 4
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %4, align 4, !tbaa !7
  %5 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %6 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  call void @_ZN7details25cm_svm_scatter_write_implIiLi1EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %5, <1 x i32> %6)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32) #4 !dbg !170 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4, !tbaa !13
  %3 = load i32, i32* %2, align 4, !dbg !171, !tbaa !13
  %4 = call <1 x i32> @_ZL11vector_castILi1EjEu2CMvbT__T0_S0_(i32 %3), !dbg !172
  ret <1 x i32> %4, !dbg !173
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i32> @_ZL11vector_castIiEu2CMvb1_T_S0_(i32) #4 !dbg !174 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4, !tbaa !13
  %3 = load i32, i32* %2, align 4, !dbg !175, !tbaa !13
  %4 = call <1 x i32> @_ZL11vector_castILi1EiEu2CMvbT__T0_S0_(i32 %3), !dbg !176
  ret <1 x i32> %4, !dbg !177
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <5 x i32> @_Z17get_auxiliary_strv() #28 comdat !dbg !178 {
  %1 = alloca <5 x i32>, align 32
  %2 = call i32 @llvm.genx.print.format.index.p0i8(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0)), !dbg !179
  %3 = load <5 x i32>, <5 x i32>* %1, align 32, !dbg !180, !tbaa !7
  %4 = insertelement <1 x i32> undef, i32 %2, i32 0, !dbg !180
  %5 = call <5 x i32> @llvm.genx.wrregioni.v5i32.v1i32.i16.i1(<5 x i32> %3, <1 x i32> %4, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !180
  store <5 x i32> %5, <5 x i32>* %1, align 32, !dbg !180, !tbaa !7
  %6 = call i32 @llvm.genx.print.format.index.p0i8(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0)), !dbg !181
  %7 = load <5 x i32>, <5 x i32>* %1, align 32, !dbg !182, !tbaa !7
  %8 = insertelement <1 x i32> undef, i32 %6, i32 0, !dbg !182
  %9 = call <5 x i32> @llvm.genx.wrregioni.v5i32.v1i32.i16.i1(<5 x i32> %7, <1 x i32> %8, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true), !dbg !182
  store <5 x i32> %9, <5 x i32>* %1, align 32, !dbg !182, !tbaa !7
  %10 = call i32 @llvm.genx.print.format.index.p0i8(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str.2, i32 0, i32 0)), !dbg !183
  %11 = load <5 x i32>, <5 x i32>* %1, align 32, !dbg !184, !tbaa !7
  %12 = insertelement <1 x i32> undef, i32 %10, i32 0, !dbg !184
  %13 = call <5 x i32> @llvm.genx.wrregioni.v5i32.v1i32.i16.i1(<5 x i32> %11, <1 x i32> %12, i32 0, i32 1, i32 0, i16 8, i32 undef, i1 true), !dbg !184
  store <5 x i32> %13, <5 x i32>* %1, align 32, !dbg !184, !tbaa !7
  %14 = call i32 @llvm.genx.print.format.index.p0i8(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.3, i32 0, i32 0)), !dbg !185
  %15 = load <5 x i32>, <5 x i32>* %1, align 32, !dbg !186, !tbaa !7
  %16 = insertelement <1 x i32> undef, i32 %14, i32 0, !dbg !186
  %17 = call <5 x i32> @llvm.genx.wrregioni.v5i32.v1i32.i16.i1(<5 x i32> %15, <1 x i32> %16, i32 0, i32 1, i32 0, i16 12, i32 undef, i1 true), !dbg !186
  store <5 x i32> %17, <5 x i32>* %1, align 32, !dbg !186, !tbaa !7
  %18 = call i32 @llvm.genx.print.format.index.p0i8(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.4, i32 0, i32 0)), !dbg !187
  %19 = load <5 x i32>, <5 x i32>* %1, align 32, !dbg !188, !tbaa !7
  %20 = insertelement <1 x i32> undef, i32 %18, i32 0, !dbg !188
  %21 = call <5 x i32> @llvm.genx.wrregioni.v5i32.v1i32.i16.i1(<5 x i32> %19, <1 x i32> %20, i32 0, i32 1, i32 0, i16 16, i32 undef, i1 true), !dbg !188
  store <5 x i32> %21, <5 x i32>* %1, align 32, !dbg !188, !tbaa !7
  %22 = load <5 x i32>, <5 x i32>* %1, align 32, !dbg !189, !tbaa !7
  ret <5 x i32> %22, !dbg !190
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo14switchEncodingI13UniformWriter13VaryingWriterEEbNS_8EncodingET_T0_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %class.VaryingWriter* byval(%class.VaryingWriter) align 32) #25 comdat !dbg !191 {
  %4 = alloca i8, align 1
  %5 = alloca %class.UniformWriter, align 32
  %6 = alloca %class.VaryingWriter, align 32
  store i8 %0, i8* %4, align 1, !tbaa !193
  %7 = load i8, i8* %4, align 1, !dbg !195, !tbaa !193
  %8 = bitcast %class.UniformWriter* %5 to i8*, !dbg !196
  %9 = bitcast %class.UniformWriter* %1 to i8*, !dbg !196
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %8, i8* align 32 %9, i32 64, i1 false), !dbg !196, !tbaa.struct !197
  %10 = call zeroext i1 @_ZN9PrintInfo22switchEncoding4UniformI13UniformWriterEEbNS_8EncodingET_(i8 signext %7, %class.UniformWriter* byval(%class.UniformWriter) align 32 %5), !dbg !198
  br i1 %10, label %16, label %11, !dbg !199

; <label>:11:                                     ; preds = %3
  %12 = load i8, i8* %4, align 1, !dbg !200, !tbaa !193
  %13 = bitcast %class.VaryingWriter* %6 to i8*, !dbg !201
  %14 = bitcast %class.VaryingWriter* %2 to i8*, !dbg !201
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %13, i8* align 32 %14, i32 64, i1 false), !dbg !201, !tbaa.struct !197
  %15 = call zeroext i1 @_ZN9PrintInfo22switchEncoding4VaryingI13VaryingWriterEEbNS_8EncodingET_(i8 signext %12, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %6), !dbg !202
  br label %16, !dbg !199

; <label>:16:                                     ; preds = %11, %3
  %17 = phi i1 [ true, %3 ], [ %15, %11 ]
  ret i1 %17, !dbg !203
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter*, i32* dereferenceable(4), i32** dereferenceable(4), i32, i64, <5 x i32>) unnamed_addr #28 comdat align 2 !dbg !204 {
  %7 = alloca %class.UniformWriter*, align 4
  %8 = alloca i32*, align 4
  %9 = alloca i32**, align 4
  %10 = alloca i32, align 4
  %11 = alloca i64, align 8
  %12 = alloca <5 x i32>, align 32
  store %class.UniformWriter* %0, %class.UniformWriter** %7, align 4, !tbaa !17
  store i32* %1, i32** %8, align 4, !tbaa !17
  store i32** %2, i32*** %9, align 4, !tbaa !17
  store i32 %3, i32* %10, align 4, !tbaa !13
  store i64 %4, i64* %11, align 8, !tbaa !19
  store <5 x i32> %5, <5 x i32>* %12, align 32, !tbaa !7
  %13 = load %class.UniformWriter*, %class.UniformWriter** %7, align 4
  %14 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %13, i32 0, i32 0, !dbg !205
  %15 = load i32*, i32** %8, align 4, !dbg !206, !tbaa !17
  store i32* %15, i32** %14, align 32, !dbg !205, !tbaa !17
  %16 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %13, i32 0, i32 1, !dbg !207
  %17 = load i32**, i32*** %9, align 4, !dbg !208, !tbaa !17
  store i32** %17, i32*** %16, align 4, !dbg !207, !tbaa !17
  %18 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %13, i32 0, i32 2, !dbg !209
  %19 = load i32, i32* %10, align 4, !dbg !210, !tbaa !13
  store i32 %19, i32* %18, align 8, !dbg !209, !tbaa !211
  %20 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %13, i32 0, i32 3, !dbg !213
  %21 = load i64, i64* %11, align 8, !dbg !214, !tbaa !19
  store i64 %21, i64* %20, align 16, !dbg !213, !tbaa !215
  %22 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %13, i32 0, i32 4, !dbg !216
  %23 = load <5 x i32>, <5 x i32>* %12, align 32, !dbg !217, !tbaa !7
  store <5 x i32> %23, <5 x i32>* %22, align 32, !dbg !216, !tbaa !218
  ret void, !dbg !219
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriterC2ERjRPKjiyu2CMvb5_i(%class.VaryingWriter*, i32* dereferenceable(4), i32** dereferenceable(4), i32, i64, <5 x i32>) unnamed_addr #28 comdat align 2 !dbg !220 {
  %7 = alloca %class.VaryingWriter*, align 4
  %8 = alloca i32*, align 4
  %9 = alloca i32**, align 4
  %10 = alloca i32, align 4
  %11 = alloca i64, align 8
  %12 = alloca <5 x i32>, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %7, align 4, !tbaa !17
  store i32* %1, i32** %8, align 4, !tbaa !17
  store i32** %2, i32*** %9, align 4, !tbaa !17
  store i32 %3, i32* %10, align 4, !tbaa !13
  store i64 %4, i64* %11, align 8, !tbaa !19
  store <5 x i32> %5, <5 x i32>* %12, align 32, !tbaa !7
  %13 = load %class.VaryingWriter*, %class.VaryingWriter** %7, align 4
  %14 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %13, i32 0, i32 0, !dbg !221
  %15 = load i32*, i32** %8, align 4, !dbg !222, !tbaa !17
  store i32* %15, i32** %14, align 32, !dbg !221, !tbaa !17
  %16 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %13, i32 0, i32 1, !dbg !223
  %17 = load i32**, i32*** %9, align 4, !dbg !224, !tbaa !17
  store i32** %17, i32*** %16, align 4, !dbg !223, !tbaa !17
  %18 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %13, i32 0, i32 2, !dbg !225
  %19 = load i32, i32* %10, align 4, !dbg !226, !tbaa !13
  store i32 %19, i32* %18, align 8, !dbg !225, !tbaa !227
  %20 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %13, i32 0, i32 3, !dbg !229
  %21 = load i64, i64* %11, align 8, !dbg !230, !tbaa !19
  store i64 %21, i64* %20, align 16, !dbg !229, !tbaa !231
  %22 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %13, i32 0, i32 4, !dbg !232
  %23 = load <5 x i32>, <5 x i32>* %12, align 32, !dbg !233, !tbaa !7
  store <5 x i32> %23, <5 x i32>* %22, align 32, !dbg !232, !tbaa !234
  ret void, !dbg !235
}

; Function Attrs: noinline nounwind
define internal i32 @__num_cores() #29 !dbg !236 {
  ret i32 -1, !dbg !237
}

; Function Attrs: nounwind readnone
declare <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32>, <1 x i32>, i32, i32, i32, i16, i32, i1) #2

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_Z12write_atomicIL14CmAtomicOpType0EjLi8EENSt9enable_ifIXaaaaaaaaaaneT_LS0_7EneT_LS0_18EneT_LS0_2EneT_LS0_3EneT_LS0_255EclL_ZN7detailsL10isPowerOf2EjjET1_Li32EEEvE4typeE15cm_surfaceindexu2CMvbT1__ju2CMvbT1__T0_u2CMvrT1__S6_(i32, <8 x i32>, <8 x i32>, <8 x i32>*) #10 comdat {
  %5 = alloca i32, align 4
  %6 = alloca <8 x i32>, align 32
  %7 = alloca <8 x i32>, align 32
  %8 = alloca <8 x i32>*, align 32
  %9 = alloca <8 x i16>, align 16
  %10 = alloca <8 x i32>, align 32
  %11 = alloca <8 x i32>*, align 32
  store i32 %0, i32* %5, align 4, !tbaa !31
  store <8 x i32> %1, <8 x i32>* %6, align 32, !tbaa !7
  store <8 x i32> %2, <8 x i32>* %7, align 32, !tbaa !7
  store <8 x i32>* %3, <8 x i32>** %8, align 32, !tbaa !7
  call void @_ZN7details18is_valid_atomic_opIjL14CmAtomicOpType0ELi8EE5checkEv()
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %9, align 16, !tbaa !7
  %12 = load <8 x i16>, <8 x i16>* %9, align 16, !tbaa !7
  %13 = load i32, i32* %5, align 4, !tbaa !31
  %14 = load <8 x i32>, <8 x i32>* %6, align 32, !tbaa !7
  %15 = load <8 x i32>, <8 x i32>* %7, align 32, !tbaa !7
  %16 = load <8 x i32>, <8 x i32>* %10, align 32, !tbaa !7
  %17 = call <8 x i32> @llvm.genx.vload.v8i32.p0v8i32(<8 x i32>* %3)
  %18 = trunc <8 x i16> %12 to <8 x i1>
  %19 = mul <8 x i32> %14, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %20 = call <8 x i32> @llvm.genx.dword.atomic.add.v8i32.v8i1.v8i32(<8 x i1> %18, i32 %13, <8 x i32> %19, <8 x i32> %15, <8 x i32> %17)
  call void @llvm.genx.vstore.v8i32.p0v8i32(<8 x i32> %20, <8 x i32>* %3)
  ret void
}

; Function Attrs: nounwind readnone
declare <1 x i32> @llvm.genx.rdregioni.v1i32.v8i32.i16(<8 x i32>, i32, i32, i32, i16, i32) #2

; Function Attrs: noinline nounwind
define internal void @_ZN7details18is_valid_atomic_opIjL14CmAtomicOpType0ELi8EE5checkEv() #30 comdat align 2 !dbg !239 {
  ret void, !dbg !241
}

declare dso_local <8 x i32> @_ZN7details32__cm_intrinsic_impl_atomic_writeIL14CmAtomicOpType0ELi8EjEEu2CMvbT0__T1_u2CMvbT0__t15cm_surfaceindexu2CMvbT0__jS3_S3_S3_(<8 x i16>, i32, <8 x i32>, <8 x i32>, <8 x i32>, <8 x i32>) #24

; Function Attrs: nounwind
declare <8 x i32> @llvm.genx.vload.v8i32.p0v8i32(<8 x i32>*) #27

; Function Attrs: nounwind
declare <8 x i32> @llvm.genx.dword.atomic.add.v8i32.v8i1.v8i32(<8 x i1>, i32, <8 x i32>, <8 x i32>, <8 x i32>) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v8i32.p0v8i32(<8 x i32>, <8 x i32>*) #27

declare dso_local void @_Z13cm_svm_atomicIjLi8EEv14CmAtomicOpTypeu2CMvbT0__ju2CMvrT0__T_u2CMvbT0__S2_(i32, <8 x i32>, <8 x i32>*, <8 x i32>) #24

; Function Attrs: nounwind
declare <8 x i32> @llvm.genx.svm.atomic.add.v8i32.v8i1.v8i32(<8 x i1>, <8 x i32>, <8 x i32>, <8 x i32>) #27

; Function Attrs: nounwind readnone
declare i32 @llvm.genx.print.format.index.p0i8(i8*) #2

; Function Attrs: nounwind readnone
declare <5 x i32> @llvm.genx.wrregioni.v5i32.v1i32.i16.i1(<5 x i32>, <1 x i32>, i32, i32, i32, i16, i32, i1) #2

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v2i8.p0v2i8(<2 x i8>, <2 x i8>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v4i8.p0v4i8(<4 x i8>, <4 x i8>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v8i8.p0v8i8(<8 x i8>, <8 x i8>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v16i8.p0v16i8(<16 x i8>, <16 x i8>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v32i8.p0v32i8(<32 x i8>, <32 x i8>*) #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal float @_ZN7details16__impl_hex2floatEj(i32) #25 comdat {
  %2 = alloca i32, align 4
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x float>, align 4
  store i32 %0, i32* %2, align 4, !tbaa !13
  %5 = load i32, i32* %2, align 4, !tbaa !13
  %6 = insertelement <1 x i32> undef, i32 %5, i32 0
  %7 = shufflevector <1 x i32> %6, <1 x i32> undef, <1 x i32> zeroinitializer
  store <1 x i32> %7, <1 x i32>* %3, align 4, !tbaa !7
  %8 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %9 = bitcast <1 x i32> %8 to <1 x float>
  store <1 x float> %9, <1 x float>* %4, align 4, !tbaa !7
  %10 = load <1 x float>, <1 x float>* %4, align 4, !tbaa !7
  %11 = call <1 x float> @llvm.genx.rdregionf.v1f32.v1f32.i16(<1 x float> %10, i32 0, i32 1, i32 0, i16 0, i32 undef)
  %12 = extractelement <1 x float> %11, i32 0
  ret float %12
}

; Function Attrs: nounwind readnone
declare <1 x float> @llvm.genx.rdregionf.v1f32.v1f32.i16(<1 x float>, i32, i32, i32, i16, i32) #2

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v2i16.p0v2i16(<2 x i16>, <2 x i16>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v4i16.p0v4i16(<4 x i16>, <4 x i16>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v8i16.p0v8i16(<8 x i16>, <8 x i16>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v16i16.p0v16i16(<16 x i16>, <16 x i16>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v32i16.p0v32i16(<32 x i16>, <32 x i16>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v2i32.p0v2i32(<2 x i32>, <2 x i32>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v4i32.p0v4i32(<4 x i32>, <4 x i32>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v16i32.p0v16i32(<16 x i32>, <16 x i32>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v32i32.p0v32i32(<32 x i32>, <32 x i32>*) #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z13CopyPlainTextILj128EEiPKcu2CMvrT__cii(i8*, <128 x i8>*, i32, i32) #25 comdat !dbg !248 {
  %5 = alloca i8*, align 4
  %6 = alloca <128 x i8>*, align 128
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store i8* %0, i8** %5, align 4, !tbaa !17
  store <128 x i8>* %1, <128 x i8>** %6, align 128, !tbaa !7
  store i32 %2, i32* %7, align 4, !tbaa !13
  store i32 %3, i32* %8, align 4, !tbaa !13
  %9 = load i8*, i8** %5, align 4, !dbg !249, !tbaa !17
  %10 = load i32, i32* %7, align 4, !dbg !250, !tbaa !13
  %11 = load i32, i32* %8, align 4, !dbg !251, !tbaa !13
  %12 = call i32 @_ZN7details11CopyTillSepIJLc37ELc0EEPKcLj128EEEiT0_iu2CMvrT1__cii(i8* %9, i32 0, <128 x i8>* %1, i32 %10, i32 %11), !dbg !252
  ret i32 %12, !dbg !253
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN7details7Arg2StrI9ArgWriterEEu2CMvb100_ccRT_(i8 signext, %class.ArgWriter* dereferenceable(24)) #31 comdat !dbg !254 {
  %3 = alloca i8, align 1
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>, align 128
  store i8 %0, i8* %3, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %4, align 4, !tbaa !17
  %6 = load i8, i8* %3, align 1, !dbg !255, !tbaa !7
  %7 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4, !dbg !256, !tbaa !17
  %8 = call zeroext i1 @_ZN7details17Arg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %6, %class.ArgWriter* dereferenceable(24) %7, <100 x i8>* %5), !dbg !257
  br i1 %8, label %37, label %9, !dbg !258

; <label>:9:                                      ; preds = %2
  %10 = load i8, i8* %3, align 1, !dbg !259, !tbaa !7
  %11 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4, !dbg !260, !tbaa !17
  %12 = call zeroext i1 @_ZN7details17Arg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %10, %class.ArgWriter* dereferenceable(24) %11, <100 x i8>* %5), !dbg !261
  br i1 %12, label %37, label %13, !dbg !262

; <label>:13:                                     ; preds = %9
  %14 = load i8, i8* %3, align 1, !dbg !263, !tbaa !7
  %15 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4, !dbg !264, !tbaa !17
  %16 = call zeroext i1 @_ZN7details17Arg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %14, %class.ArgWriter* dereferenceable(24) %15, <100 x i8>* %5), !dbg !265
  br i1 %16, label %37, label %17, !dbg !266

; <label>:17:                                     ; preds = %13
  %18 = load i8, i8* %3, align 1, !dbg !267, !tbaa !7
  %19 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4, !dbg !268, !tbaa !17
  %20 = call zeroext i1 @_ZN7details17Arg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %18, %class.ArgWriter* dereferenceable(24) %19, <100 x i8>* %5), !dbg !269
  br i1 %20, label %37, label %21, !dbg !270

; <label>:21:                                     ; preds = %17
  %22 = load i8, i8* %3, align 1, !dbg !271, !tbaa !7
  %23 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4, !dbg !272, !tbaa !17
  %24 = call zeroext i1 @_ZN7details17Arg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %22, %class.ArgWriter* dereferenceable(24) %23, <100 x i8>* %5), !dbg !273
  br i1 %24, label %37, label %25, !dbg !274

; <label>:25:                                     ; preds = %21
  %26 = load i8, i8* %3, align 1, !dbg !275, !tbaa !7
  %27 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4, !dbg !276, !tbaa !17
  %28 = call zeroext i1 @_ZN7details17Arg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %26, %class.ArgWriter* dereferenceable(24) %27, <100 x i8>* %5), !dbg !277
  br i1 %28, label %37, label %29, !dbg !278

; <label>:29:                                     ; preds = %25
  %30 = load i8, i8* %3, align 1, !dbg !279, !tbaa !7
  %31 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4, !dbg !280, !tbaa !17
  %32 = call zeroext i1 @_ZN7details17Arg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %30, %class.ArgWriter* dereferenceable(24) %31, <100 x i8>* %5), !dbg !281
  br i1 %32, label %37, label %33, !dbg !282

; <label>:33:                                     ; preds = %29
  %34 = load i8, i8* %3, align 1, !dbg !283, !tbaa !7
  %35 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4, !dbg !284, !tbaa !17
  %36 = call zeroext i1 @_ZN7details17Arg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %34, %class.ArgWriter* dereferenceable(24) %35, <100 x i8>* %5), !dbg !285
  br label %37, !dbg !282

; <label>:37:                                     ; preds = %33, %29, %25, %21, %17, %13, %9, %2
  %38 = phi i1 [ true, %29 ], [ true, %25 ], [ true, %21 ], [ true, %17 ], [ true, %13 ], [ true, %9 ], [ true, %2 ], [ %36, %33 ]
  %39 = load <100 x i8>, <100 x i8>* %5, align 128, !dbg !286, !tbaa !7
  ret <100 x i8> %39, !dbg !287
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z12CopyFullTextILj100ELj128EEiu2CMvrT__ciu2CMvrT0__cii(<100 x i8>*, i32, <128 x i8>*, i32, i32) #25 comdat !dbg !288 {
  %6 = alloca <100 x i8>*, align 128
  %7 = alloca i32, align 4
  %8 = alloca <128 x i8>*, align 128
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  store <100 x i8>* %0, <100 x i8>** %6, align 128, !tbaa !7
  store i32 %1, i32* %7, align 4, !tbaa !13
  store <128 x i8>* %2, <128 x i8>** %8, align 128, !tbaa !7
  store i32 %3, i32* %9, align 4, !tbaa !13
  store i32 %4, i32* %10, align 4, !tbaa !13
  %11 = load i32, i32* %7, align 4, !dbg !289, !tbaa !13
  %12 = load i32, i32* %9, align 4, !dbg !290, !tbaa !13
  %13 = load i32, i32* %10, align 4, !dbg !291, !tbaa !13
  %14 = call i32 @_ZN7details11CopyTillSepIJLc0EELj100ELj128EEEiu2CMvrT0__ciu2CMvrT1__cii(<100 x i8>* %0, i32 %11, <128 x i8>* %2, i32 %12, i32 %13), !dbg !292
  ret i32 %14, !dbg !293
}

; Function Attrs: nounwind readnone
declare <128 x i8> @llvm.genx.wrregioni.v128i8.v1i8.i16.i1(<128 x i8>, <1 x i8>, i32, i32, i32, i16, i32, i1) #2

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details11CopyTillSepIJLc37ELc0EEPKcLj128EEEiT0_iu2CMvrT1__cii(i8*, i32, <128 x i8>*, i32, i32) #25 comdat !dbg !294 {
  %6 = alloca i8*, align 4
  %7 = alloca i32, align 4
  %8 = alloca <128 x i8>*, align 128
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store i8* %0, i8** %6, align 4, !tbaa !17
  store i32 %1, i32* %7, align 4, !tbaa !13
  store <128 x i8>* %2, <128 x i8>** %8, align 128, !tbaa !7
  store i32 %3, i32* %9, align 4, !tbaa !13
  store i32 %4, i32* %10, align 4, !tbaa !13
  %12 = load i32, i32* %9, align 4, !dbg !295, !tbaa !13
  store i32 %12, i32* %11, align 4, !dbg !296, !tbaa !13
  br label %13, !dbg !297

; <label>:13:                                     ; preds = %32, %5
  %14 = load i8*, i8** %6, align 4, !dbg !298, !tbaa !17
  %15 = load i32, i32* %7, align 4, !dbg !299, !tbaa !13
  %16 = getelementptr inbounds i8, i8* %14, i32 %15, !dbg !298
  %17 = load i8, i8* %16, align 1, !dbg !298, !tbaa !7
  %18 = sext i8 %17 to i32, !dbg !298
  %19 = icmp ne i32 %18, 37, !dbg !300
  br i1 %19, label %20, label %30, !dbg !301

; <label>:20:                                     ; preds = %13
  %21 = load i8*, i8** %6, align 4, !dbg !298, !tbaa !17
  %22 = load i32, i32* %7, align 4, !dbg !299, !tbaa !13
  %23 = getelementptr inbounds i8, i8* %21, i32 %22, !dbg !298
  %24 = load i8, i8* %23, align 1, !dbg !298, !tbaa !7
  %25 = sext i8 %24 to i32, !dbg !298
  %26 = icmp ne i32 %25, 0, !dbg !300
  br i1 %26, label %27, label %30, !dbg !302

; <label>:27:                                     ; preds = %20
  %28 = load i32, i32* %10, align 4, !dbg !303, !tbaa !13
  %29 = icmp ne i32 %28, 0, !dbg !303
  br label %30

; <label>:30:                                     ; preds = %27, %20, %13
  %31 = phi i1 [ false, %20 ], [ false, %13 ], [ %29, %27 ], !dbg !304
  br i1 %31, label %32, label %47, !dbg !297

; <label>:32:                                     ; preds = %30
  %33 = load i8*, i8** %6, align 4, !dbg !305, !tbaa !17
  %34 = load i32, i32* %7, align 4, !dbg !306, !tbaa !13
  %35 = add nsw i32 %34, 1, !dbg !306
  store i32 %35, i32* %7, align 4, !dbg !306, !tbaa !13
  %36 = getelementptr inbounds i8, i8* %33, i32 %34, !dbg !305
  %37 = load i8, i8* %36, align 1, !dbg !305, !tbaa !7
  %38 = load i32, i32* %9, align 4, !dbg !307, !tbaa !13
  %39 = add nsw i32 %38, 1, !dbg !307
  store i32 %39, i32* %9, align 4, !dbg !307, !tbaa !13
  %40 = trunc i32 %38 to i16, !dbg !308
  %41 = call <128 x i8> @llvm.genx.vload.v128i8.p0v128i8(<128 x i8>* %2), !dbg !309
  %42 = insertelement <1 x i8> undef, i8 %37, i32 0, !dbg !309
  %43 = mul i16 %40, 1, !dbg !309
  %44 = call <128 x i8> @llvm.genx.wrregioni.v128i8.v1i8.i16.i1(<128 x i8> %41, <1 x i8> %42, i32 0, i32 1, i32 0, i16 %43, i32 undef, i1 true), !dbg !309
  call void @llvm.genx.vstore.v128i8.p0v128i8(<128 x i8> %44, <128 x i8>* %2), !dbg !309
  %45 = load i32, i32* %10, align 4, !dbg !310, !tbaa !13
  %46 = add nsw i32 %45, -1, !dbg !310
  store i32 %46, i32* %10, align 4, !dbg !310, !tbaa !13
  br label %13, !dbg !297, !llvm.loop !311

; <label>:47:                                     ; preds = %30
  %48 = load i32, i32* %9, align 4, !dbg !313, !tbaa !13
  %49 = load i32, i32* %11, align 4, !dbg !314, !tbaa !13
  %50 = sub nsw i32 %48, %49, !dbg !315
  ret i32 %50, !dbg !316
}

; Function Attrs: nounwind
declare <128 x i8> @llvm.genx.vload.v128i8.p0v128i8(<128 x i8>*) #27

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v128i8.p0v128i8(<128 x i8>, <128 x i8>*) #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details17Arg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #25 comdat !dbg !317 {
  %4 = alloca i8, align 1
  %5 = alloca %class.ArgWriter*, align 4
  %6 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %4, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %5, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %6, align 128, !tbaa !7
  %7 = load i8, i8* %4, align 1, !dbg !318, !tbaa !7
  %8 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !319, !tbaa !17
  %9 = call zeroext i1 @_ZN7details20UniArg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %7, %class.ArgWriter* dereferenceable(24) %8, <100 x i8>* %2), !dbg !320
  br i1 %9, label %14, label %10, !dbg !321

; <label>:10:                                     ; preds = %3
  %11 = load i8, i8* %4, align 1, !dbg !322, !tbaa !7
  %12 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !323, !tbaa !17
  %13 = call zeroext i1 @_ZN7details20VarArg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %11, %class.ArgWriter* dereferenceable(24) %12, <100 x i8>* %2), !dbg !324
  br label %14, !dbg !321

; <label>:14:                                     ; preds = %10, %3
  %15 = phi i1 [ true, %3 ], [ %13, %10 ]
  ret i1 %15, !dbg !325
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details17Arg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #25 comdat !dbg !326 {
  %4 = alloca i8, align 1
  %5 = alloca %class.ArgWriter*, align 4
  %6 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %4, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %5, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %6, align 128, !tbaa !7
  %7 = load i8, i8* %4, align 1, !dbg !327, !tbaa !7
  %8 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !328, !tbaa !17
  %9 = call zeroext i1 @_ZN7details20UniArg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %7, %class.ArgWriter* dereferenceable(24) %8, <100 x i8>* %2), !dbg !329
  br i1 %9, label %14, label %10, !dbg !330

; <label>:10:                                     ; preds = %3
  %11 = load i8, i8* %4, align 1, !dbg !331, !tbaa !7
  %12 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !332, !tbaa !17
  %13 = call zeroext i1 @_ZN7details20VarArg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %11, %class.ArgWriter* dereferenceable(24) %12, <100 x i8>* %2), !dbg !333
  br label %14, !dbg !330

; <label>:14:                                     ; preds = %10, %3
  %15 = phi i1 [ true, %3 ], [ %13, %10 ]
  ret i1 %15, !dbg !334
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details17Arg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #25 comdat !dbg !335 {
  %4 = alloca i8, align 1
  %5 = alloca %class.ArgWriter*, align 4
  %6 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %4, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %5, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %6, align 128, !tbaa !7
  %7 = load i8, i8* %4, align 1, !dbg !336, !tbaa !7
  %8 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !337, !tbaa !17
  %9 = call zeroext i1 @_ZN7details20UniArg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %7, %class.ArgWriter* dereferenceable(24) %8, <100 x i8>* %2), !dbg !338
  br i1 %9, label %14, label %10, !dbg !339

; <label>:10:                                     ; preds = %3
  %11 = load i8, i8* %4, align 1, !dbg !340, !tbaa !7
  %12 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !341, !tbaa !17
  %13 = call zeroext i1 @_ZN7details20VarArg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %11, %class.ArgWriter* dereferenceable(24) %12, <100 x i8>* %2), !dbg !342
  br label %14, !dbg !339

; <label>:14:                                     ; preds = %10, %3
  %15 = phi i1 [ true, %3 ], [ %13, %10 ]
  ret i1 %15, !dbg !343
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details17Arg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #25 comdat !dbg !344 {
  %4 = alloca i8, align 1
  %5 = alloca %class.ArgWriter*, align 4
  %6 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %4, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %5, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %6, align 128, !tbaa !7
  %7 = load i8, i8* %4, align 1, !dbg !345, !tbaa !7
  %8 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !346, !tbaa !17
  %9 = call zeroext i1 @_ZN7details20UniArg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %7, %class.ArgWriter* dereferenceable(24) %8, <100 x i8>* %2), !dbg !347
  br i1 %9, label %14, label %10, !dbg !348

; <label>:10:                                     ; preds = %3
  %11 = load i8, i8* %4, align 1, !dbg !349, !tbaa !7
  %12 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !350, !tbaa !17
  %13 = call zeroext i1 @_ZN7details20VarArg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %11, %class.ArgWriter* dereferenceable(24) %12, <100 x i8>* %2), !dbg !351
  br label %14, !dbg !348

; <label>:14:                                     ; preds = %10, %3
  %15 = phi i1 [ true, %3 ], [ %13, %10 ]
  ret i1 %15, !dbg !352
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details17Arg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #25 comdat !dbg !353 {
  %4 = alloca i8, align 1
  %5 = alloca %class.ArgWriter*, align 4
  %6 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %4, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %5, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %6, align 128, !tbaa !7
  %7 = load i8, i8* %4, align 1, !dbg !354, !tbaa !7
  %8 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !355, !tbaa !17
  %9 = call zeroext i1 @_ZN7details20UniArg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %7, %class.ArgWriter* dereferenceable(24) %8, <100 x i8>* %2), !dbg !356
  br i1 %9, label %14, label %10, !dbg !357

; <label>:10:                                     ; preds = %3
  %11 = load i8, i8* %4, align 1, !dbg !358, !tbaa !7
  %12 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !359, !tbaa !17
  %13 = call zeroext i1 @_ZN7details20VarArg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %11, %class.ArgWriter* dereferenceable(24) %12, <100 x i8>* %2), !dbg !360
  br label %14, !dbg !357

; <label>:14:                                     ; preds = %10, %3
  %15 = phi i1 [ true, %3 ], [ %13, %10 ]
  ret i1 %15, !dbg !361
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details17Arg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #25 comdat !dbg !362 {
  %4 = alloca i8, align 1
  %5 = alloca %class.ArgWriter*, align 4
  %6 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %4, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %5, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %6, align 128, !tbaa !7
  %7 = load i8, i8* %4, align 1, !dbg !363, !tbaa !7
  %8 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !364, !tbaa !17
  %9 = call zeroext i1 @_ZN7details20UniArg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %7, %class.ArgWriter* dereferenceable(24) %8, <100 x i8>* %2), !dbg !365
  br i1 %9, label %14, label %10, !dbg !366

; <label>:10:                                     ; preds = %3
  %11 = load i8, i8* %4, align 1, !dbg !367, !tbaa !7
  %12 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !368, !tbaa !17
  %13 = call zeroext i1 @_ZN7details20VarArg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %11, %class.ArgWriter* dereferenceable(24) %12, <100 x i8>* %2), !dbg !369
  br label %14, !dbg !366

; <label>:14:                                     ; preds = %10, %3
  %15 = phi i1 [ true, %3 ], [ %13, %10 ]
  ret i1 %15, !dbg !370
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details17Arg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #25 comdat !dbg !371 {
  %4 = alloca i8, align 1
  %5 = alloca %class.ArgWriter*, align 4
  %6 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %4, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %5, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %6, align 128, !tbaa !7
  %7 = load i8, i8* %4, align 1, !dbg !372, !tbaa !7
  %8 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !373, !tbaa !17
  %9 = call zeroext i1 @_ZN7details20UniArg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %7, %class.ArgWriter* dereferenceable(24) %8, <100 x i8>* %2), !dbg !374
  br i1 %9, label %14, label %10, !dbg !375

; <label>:10:                                     ; preds = %3
  %11 = load i8, i8* %4, align 1, !dbg !376, !tbaa !7
  %12 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !377, !tbaa !17
  %13 = call zeroext i1 @_ZN7details20VarArg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %11, %class.ArgWriter* dereferenceable(24) %12, <100 x i8>* %2), !dbg !378
  br label %14, !dbg !375

; <label>:14:                                     ; preds = %10, %3
  %15 = phi i1 [ true, %3 ], [ %13, %10 ]
  ret i1 %15, !dbg !379
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details17Arg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #25 comdat !dbg !380 {
  %4 = alloca i8, align 1
  %5 = alloca %class.ArgWriter*, align 4
  %6 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %4, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %5, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %6, align 128, !tbaa !7
  %7 = load i8, i8* %4, align 1, !dbg !381, !tbaa !7
  %8 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !382, !tbaa !17
  %9 = call zeroext i1 @_ZN7details20UniArg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %7, %class.ArgWriter* dereferenceable(24) %8, <100 x i8>* %2), !dbg !383
  br i1 %9, label %14, label %10, !dbg !384

; <label>:10:                                     ; preds = %3
  %11 = load i8, i8* %4, align 1, !dbg !385, !tbaa !7
  %12 = load %class.ArgWriter*, %class.ArgWriter** %5, align 4, !dbg !386, !tbaa !17
  %13 = call zeroext i1 @_ZN7details20VarArg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext %11, %class.ArgWriter* dereferenceable(24) %12, <100 x i8>* %2), !dbg !387
  br label %14, !dbg !384

; <label>:14:                                     ; preds = %10, %3
  %15 = phi i1 [ true, %3 ], [ %13, %10 ]
  ret i1 %15, !dbg !388
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20UniArg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !389 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !390, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !390
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIbEENS_8EncodingEv(), !dbg !391
  %11 = sext i8 %10 to i32, !dbg !391
  %12 = icmp eq i32 %9, %11, !dbg !392
  br i1 %12, label %13, label %16, !dbg !390

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !393, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11uniform2StrIbEEDav(%class.ArgWriter* %14), !dbg !394
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !395
  store i1 true, i1* %4, align 1, !dbg !396
  br label %17, !dbg !396

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !397
  br label %17, !dbg !397

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !398
  ret i1 %18, !dbg !398
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20VarArg2StrIfSuitableIb9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !399 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !400, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !400
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIbEENS_8EncodingEv(), !dbg !401
  %11 = sext i8 %10 to i32, !dbg !401
  %12 = icmp eq i32 %9, %11, !dbg !402
  br i1 %12, label %13, label %16, !dbg !400

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !403, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11varying2StrIbEEDav(%class.ArgWriter* %14), !dbg !404
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !405
  store i1 true, i1* %4, align 1, !dbg !406
  br label %17, !dbg !406

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !407
  br label %17, !dbg !407

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !408
  ret i1 %18, !dbg !408
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4UniformIbEENS_8EncodingEv() #25 comdat !dbg !409 {
  ret i8 1, !dbg !410
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11uniform2StrIbEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !411 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %5 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %6 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %5), !dbg !412
  %7 = icmp ne i32 %6, 0, !dbg !412
  %8 = call i8* @_Z12ValueAdapterIbEDaT_(i1 zeroext %7), !dbg !413
  %9 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %8, <100 x i8>* %3, i32 0, i32 99), !dbg !414
  store i32 %9, i32* %4, align 4, !dbg !415, !tbaa !13
  %10 = load i32, i32* %4, align 4, !dbg !416, !tbaa !13
  %11 = trunc i32 %10 to i16, !dbg !416
  %12 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !417, !tbaa !7
  %13 = mul i16 %11, 1, !dbg !417
  %14 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %12, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %13, i32 undef, i1 true), !dbg !417
  store <100 x i8> %14, <100 x i8>* %3, align 128, !dbg !417, !tbaa !7
  %15 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !418, !tbaa !7
  ret <100 x i8> %15, !dbg !419
}

; Function Attrs: nounwind
declare void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8>, <100 x i8>*) #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8*, <100 x i8>*, i32, i32) #25 comdat !dbg !420 {
  %5 = alloca i8*, align 4
  %6 = alloca <100 x i8>*, align 128
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store i8* %0, i8** %5, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %6, align 128, !tbaa !7
  store i32 %2, i32* %7, align 4, !tbaa !13
  store i32 %3, i32* %8, align 4, !tbaa !13
  %9 = load i8*, i8** %5, align 4, !dbg !421, !tbaa !17
  %10 = load i32, i32* %7, align 4, !dbg !422, !tbaa !13
  %11 = load i32, i32* %8, align 4, !dbg !423, !tbaa !13
  %12 = call i32 @_ZN7details11CopyTillSepIJLc0EEPKcLj100EEEiT0_iu2CMvrT1__cii(i8* %9, i32 0, <100 x i8>* %1, i32 %10, i32 %11), !dbg !424
  ret i32 %12, !dbg !425
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_Z12ValueAdapterIbEDaT_(i1 zeroext) #25 comdat !dbg !426 {
  %2 = alloca i8, align 1
  %3 = zext i1 %0 to i8
  store i8 %3, i8* %2, align 1, !tbaa !427
  %4 = load i8, i8* %2, align 1, !dbg !429, !tbaa !427, !range !430
  %5 = trunc i8 %4 to i1, !dbg !429
  %6 = call i8* @_ZN7detailsL16ValueAdapterImplEb(i1 zeroext %5), !dbg !431
  ret i8* %6, !dbg !432
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter*) #25 comdat align 2 !dbg !433 {
  %2 = alloca %class.ArgWriter*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %3 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %4 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %3, i32 0, i32 1, !dbg !434
  %5 = load i32*, i32** %4, align 4, !dbg !434, !tbaa !63
  %6 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %3, i32 0, i32 2, !dbg !435
  %7 = load i32, i32* %6, align 8, !dbg !436, !tbaa !65
  %8 = add nsw i32 %7, 1, !dbg !436
  store i32 %8, i32* %6, align 8, !dbg !436, !tbaa !65
  %9 = getelementptr inbounds i32, i32* %5, i32 %7, !dbg !434
  %10 = load i32, i32* %9, align 4, !dbg !434, !tbaa !13
  ret i32 %10, !dbg !437
}

; Function Attrs: nounwind readnone
declare <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8>, <1 x i8>, i32, i32, i32, i16, i32, i1) #2

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details11CopyTillSepIJLc0EEPKcLj100EEEiT0_iu2CMvrT1__cii(i8*, i32, <100 x i8>*, i32, i32) #25 comdat !dbg !438 {
  %6 = alloca i8*, align 4
  %7 = alloca i32, align 4
  %8 = alloca <100 x i8>*, align 128
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store i8* %0, i8** %6, align 4, !tbaa !17
  store i32 %1, i32* %7, align 4, !tbaa !13
  store <100 x i8>* %2, <100 x i8>** %8, align 128, !tbaa !7
  store i32 %3, i32* %9, align 4, !tbaa !13
  store i32 %4, i32* %10, align 4, !tbaa !13
  %12 = load i32, i32* %9, align 4, !dbg !439, !tbaa !13
  store i32 %12, i32* %11, align 4, !dbg !440, !tbaa !13
  br label %13, !dbg !441

; <label>:13:                                     ; preds = %25, %5
  %14 = load i8*, i8** %6, align 4, !dbg !442, !tbaa !17
  %15 = load i32, i32* %7, align 4, !dbg !443, !tbaa !13
  %16 = getelementptr inbounds i8, i8* %14, i32 %15, !dbg !442
  %17 = load i8, i8* %16, align 1, !dbg !442, !tbaa !7
  %18 = sext i8 %17 to i32, !dbg !442
  %19 = icmp ne i32 %18, 0, !dbg !444
  br i1 %19, label %20, label %23, !dbg !445

; <label>:20:                                     ; preds = %13
  %21 = load i32, i32* %10, align 4, !dbg !446, !tbaa !13
  %22 = icmp ne i32 %21, 0, !dbg !446
  br label %23

; <label>:23:                                     ; preds = %20, %13
  %24 = phi i1 [ false, %13 ], [ %22, %20 ], !dbg !447
  br i1 %24, label %25, label %40, !dbg !441

; <label>:25:                                     ; preds = %23
  %26 = load i8*, i8** %6, align 4, !dbg !448, !tbaa !17
  %27 = load i32, i32* %7, align 4, !dbg !449, !tbaa !13
  %28 = add nsw i32 %27, 1, !dbg !449
  store i32 %28, i32* %7, align 4, !dbg !449, !tbaa !13
  %29 = getelementptr inbounds i8, i8* %26, i32 %27, !dbg !448
  %30 = load i8, i8* %29, align 1, !dbg !448, !tbaa !7
  %31 = load i32, i32* %9, align 4, !dbg !450, !tbaa !13
  %32 = add nsw i32 %31, 1, !dbg !450
  store i32 %32, i32* %9, align 4, !dbg !450, !tbaa !13
  %33 = trunc i32 %31 to i16, !dbg !451
  %34 = call <100 x i8> @llvm.genx.vload.v100i8.p0v100i8(<100 x i8>* %2), !dbg !452
  %35 = insertelement <1 x i8> undef, i8 %30, i32 0, !dbg !452
  %36 = mul i16 %33, 1, !dbg !452
  %37 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %34, <1 x i8> %35, i32 0, i32 1, i32 0, i16 %36, i32 undef, i1 true), !dbg !452
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %37, <100 x i8>* %2), !dbg !452
  %38 = load i32, i32* %10, align 4, !dbg !453, !tbaa !13
  %39 = add nsw i32 %38, -1, !dbg !453
  store i32 %39, i32* %10, align 4, !dbg !453, !tbaa !13
  br label %13, !dbg !441, !llvm.loop !454

; <label>:40:                                     ; preds = %23
  %41 = load i32, i32* %9, align 4, !dbg !456, !tbaa !13
  %42 = load i32, i32* %11, align 4, !dbg !457, !tbaa !13
  %43 = sub nsw i32 %41, %42, !dbg !458
  ret i32 %43, !dbg !459
}

; Function Attrs: nounwind
declare <100 x i8> @llvm.genx.vload.v100i8.p0v100i8(<100 x i8>*) #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_ZN7detailsL16ValueAdapterImplEb(i1 zeroext) #25 !dbg !460 {
  %2 = alloca i8, align 1
  %3 = zext i1 %0 to i8
  store i8 %3, i8* %2, align 1, !tbaa !427
  %4 = load i8, i8* %2, align 1, !dbg !461, !tbaa !427, !range !430
  %5 = trunc i8 %4 to i1, !dbg !461
  %6 = zext i1 %5 to i64, !dbg !461
  %7 = select i1 %5, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.3, i32 0, i32 0), !dbg !461
  ret i8* %7, !dbg !462
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4VaryingIbEENS_8EncodingEv() #25 comdat !dbg !463 {
  ret i8 9, !dbg !464
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11varying2StrIbEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !465 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !466, !tbaa !7
  %8 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %7, <1 x i8> <i8 91>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !466
  store <100 x i8> %8, <100 x i8>* %3, align 128, !dbg !466, !tbaa !7
  store i32 1, i32* %4, align 4, !dbg !467, !tbaa !13
  store i32 0, i32* %5, align 4, !dbg !468, !tbaa !13
  br label %9, !dbg !469

; <label>:9:                                      ; preds = %58, %1
  %10 = load i32, i32* %5, align 4, !dbg !470, !tbaa !13
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !471
  %12 = load i32, i32* %11, align 4, !dbg !471, !tbaa !68
  %13 = icmp slt i32 %10, %12, !dbg !472
  br i1 %13, label %14, label %61, !dbg !473

; <label>:14:                                     ; preds = %9
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 4, !dbg !474
  %16 = load i64, i64* %15, align 8, !dbg !474, !tbaa !71
  %17 = load i32, i32* %5, align 4, !dbg !475, !tbaa !13
  %18 = zext i32 %17 to i64, !dbg !476
  %19 = shl i64 1, %18, !dbg !476
  %20 = and i64 %16, %19, !dbg !477
  %21 = icmp ne i64 %20, 0, !dbg !474
  br i1 %21, label %22, label %32, !dbg !474

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %4, align 4, !dbg !478, !tbaa !13
  %24 = call i32 @_Z21requiredSpace4VecElemIbL9LaneState1EEiv(), !dbg !479
  %25 = sub i32 100, %24, !dbg !480
  %26 = sub i32 %25, 1, !dbg !481
  %27 = icmp uge i32 %23, %26, !dbg !482
  br i1 %27, label %28, label %29, !dbg !478

; <label>:28:                                     ; preds = %22
  br label %61, !dbg !483

; <label>:29:                                     ; preds = %22
  %30 = load i32, i32* %4, align 4, !dbg !484, !tbaa !13
  %31 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIbL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %30), !dbg !485
  store i32 %31, i32* %4, align 4, !dbg !486, !tbaa !13
  br label %42, !dbg !487

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %4, align 4, !dbg !488, !tbaa !13
  %34 = call i32 @_Z21requiredSpace4VecElemIbL9LaneState0EEiv(), !dbg !489
  %35 = sub i32 100, %34, !dbg !490
  %36 = sub i32 %35, 1, !dbg !491
  %37 = icmp uge i32 %33, %36, !dbg !492
  br i1 %37, label %38, label %39, !dbg !488

; <label>:38:                                     ; preds = %32
  br label %61, !dbg !493

; <label>:39:                                     ; preds = %32
  %40 = load i32, i32* %4, align 4, !dbg !494, !tbaa !13
  %41 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIbL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %40), !dbg !495
  store i32 %41, i32* %4, align 4, !dbg !496, !tbaa !13
  br label %42

; <label>:42:                                     ; preds = %39, %29
  %43 = load i32, i32* %5, align 4, !dbg !497, !tbaa !13
  %44 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !498
  %45 = load i32, i32* %44, align 4, !dbg !498, !tbaa !68
  %46 = sub nsw i32 %45, 1, !dbg !499
  %47 = icmp eq i32 %43, %46, !dbg !500
  %48 = zext i1 %47 to i64, !dbg !497
  %49 = select i1 %47, i8 93, i8 44, !dbg !497
  %50 = load i32, i32* %4, align 4, !dbg !501, !tbaa !13
  %51 = trunc i32 %50 to i16, !dbg !501
  %52 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !502, !tbaa !7
  %53 = insertelement <1 x i8> undef, i8 %49, i32 0, !dbg !502
  %54 = mul i16 %51, 1, !dbg !502
  %55 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %52, <1 x i8> %53, i32 0, i32 1, i32 0, i16 %54, i32 undef, i1 true), !dbg !502
  store <100 x i8> %55, <100 x i8>* %3, align 128, !dbg !502, !tbaa !7
  %56 = load i32, i32* %4, align 4, !dbg !503, !tbaa !13
  %57 = add nsw i32 %56, 1, !dbg !503
  store i32 %57, i32* %4, align 4, !dbg !503, !tbaa !13
  call void @_ZN9ArgWriter8WriteArgIbEEvv(%class.ArgWriter* %6), !dbg !504
  br label %58, !dbg !505

; <label>:58:                                     ; preds = %42
  %59 = load i32, i32* %5, align 4, !dbg !506, !tbaa !13
  %60 = add nsw i32 %59, 1, !dbg !506
  store i32 %60, i32* %5, align 4, !dbg !506, !tbaa !13
  br label %9, !dbg !473, !llvm.loop !507

; <label>:61:                                     ; preds = %38, %28, %9
  %62 = load i32, i32* %4, align 4, !dbg !508, !tbaa !13
  %63 = trunc i32 %62 to i16, !dbg !508
  %64 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !509, !tbaa !7
  %65 = mul i16 %63, 1, !dbg !509
  %66 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %64, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %65, i32 undef, i1 true), !dbg !509
  store <100 x i8> %66, <100 x i8>* %3, align 128, !dbg !509, !tbaa !7
  %67 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !510, !tbaa !7
  ret <100 x i8> %67, !dbg !511
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIbL9LaneState1EEiv() #25 comdat !dbg !512 {
  %1 = call i8* @_Z12ValueAdapterIbEDaT_(i1 zeroext false), !dbg !513
  %2 = call i32 @_Z6strLenIPKcEiT_(i8* %1), !dbg !514
  %3 = call i8* @_Z12ValueAdapterIbEDaT_(i1 zeroext true), !dbg !515
  %4 = call i32 @_Z6strLenIPKcEiT_(i8* %3), !dbg !516
  %5 = call i32 @_Z3maxIiET_S0_S0_(i32 %2, i32 %4), !dbg !517
  ret i32 %5, !dbg !518
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIbL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !519 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %7 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %8 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %7), !dbg !520
  %9 = icmp ne i32 %8, 0, !dbg !520
  %10 = call i8* @_Z12ValueAdapterIbEDaT_(i1 zeroext %9), !dbg !521
  %11 = load i32, i32* %6, align 4, !dbg !522, !tbaa !13
  %12 = load i32, i32* %6, align 4, !dbg !523, !tbaa !13
  %13 = sub i32 100, %12, !dbg !524
  %14 = sub i32 %13, 1, !dbg !525
  %15 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %10, <100 x i8>* %1, i32 %11, i32 %14), !dbg !526
  %16 = load i32, i32* %6, align 4, !dbg !527, !tbaa !13
  %17 = add nsw i32 %16, %15, !dbg !527
  store i32 %17, i32* %6, align 4, !dbg !527, !tbaa !13
  %18 = load i32, i32* %6, align 4, !dbg !528, !tbaa !13
  ret i32 %18, !dbg !529
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIbL9LaneState0EEiv() #25 comdat !dbg !530 {
  %1 = call i32 @_Z6strLenIPKcEiT_(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @_ZL14OffLaneBoolStr, i32 0, i32 0)), !dbg !531
  ret i32 %1, !dbg !532
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIbL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !533 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %7 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %8 = load i32, i32* %6, align 4, !dbg !534, !tbaa !13
  %9 = load i32, i32* %6, align 4, !dbg !535, !tbaa !13
  %10 = sub i32 100, %9, !dbg !536
  %11 = sub i32 %10, 1, !dbg !537
  %12 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @_ZL14OffLaneBoolStr, i32 0, i32 0), <100 x i8>* %1, i32 %8, i32 %11), !dbg !538
  %13 = load i32, i32* %6, align 4, !dbg !539, !tbaa !13
  %14 = add nsw i32 %13, %12, !dbg !539
  store i32 %14, i32* %6, align 4, !dbg !539, !tbaa !13
  %15 = load i32, i32* %6, align 4, !dbg !540, !tbaa !13
  ret i32 %15, !dbg !541
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriter8WriteArgIbEEvv(%class.ArgWriter*) #25 comdat align 2 !dbg !542 {
  %2 = alloca %class.ArgWriter*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %3 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  ret void, !dbg !543
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z3maxIiET_S0_S0_(i32, i32) #25 comdat !dbg !544 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, i32* %4, align 4, !tbaa !13
  store i32 %1, i32* %5, align 4, !tbaa !13
  %6 = load i32, i32* %4, align 4, !dbg !545, !tbaa !13
  %7 = load i32, i32* %5, align 4, !dbg !546, !tbaa !13
  %8 = icmp sgt i32 %6, %7, !dbg !547
  br i1 %8, label %9, label %11, !dbg !545

; <label>:9:                                      ; preds = %2
  %10 = load i32, i32* %4, align 4, !dbg !548, !tbaa !13
  store i32 %10, i32* %3, align 4, !dbg !549
  br label %13, !dbg !549

; <label>:11:                                     ; preds = %2
  %12 = load i32, i32* %5, align 4, !dbg !550, !tbaa !13
  store i32 %12, i32* %3, align 4, !dbg !551
  br label %13, !dbg !551

; <label>:13:                                     ; preds = %11, %9
  %14 = load i32, i32* %3, align 4, !dbg !552
  ret i32 %14, !dbg !552
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z6strLenIPKcEiT_(i8*) #25 comdat !dbg !553 {
  %2 = alloca i8*, align 4
  %3 = alloca i32, align 4
  store i8* %0, i8** %2, align 4, !tbaa !17
  store i32 0, i32* %3, align 4, !dbg !554, !tbaa !13
  br label %4, !dbg !555

; <label>:4:                                      ; preds = %12, %1
  %5 = load i8*, i8** %2, align 4, !dbg !556, !tbaa !17
  %6 = load i32, i32* %3, align 4, !dbg !557, !tbaa !13
  %7 = getelementptr inbounds i8, i8* %5, i32 %6, !dbg !556
  %8 = load i8, i8* %7, align 1, !dbg !556, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !556
  %10 = icmp ne i32 %9, 0, !dbg !558
  br i1 %10, label %11, label %15, !dbg !555

; <label>:11:                                     ; preds = %4
  br label %12, !dbg !555

; <label>:12:                                     ; preds = %11
  %13 = load i32, i32* %3, align 4, !dbg !559, !tbaa !13
  %14 = add nsw i32 %13, 1, !dbg !559
  store i32 %14, i32* %3, align 4, !dbg !559, !tbaa !13
  br label %4, !dbg !555, !llvm.loop !560

; <label>:15:                                     ; preds = %4
  %16 = load i32, i32* %3, align 4, !dbg !562, !tbaa !13
  ret i32 %16, !dbg !563
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20UniArg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !564 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !565, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !565
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIiEENS_8EncodingEv(), !dbg !566
  %11 = sext i8 %10 to i32, !dbg !566
  %12 = icmp eq i32 %9, %11, !dbg !567
  br i1 %12, label %13, label %16, !dbg !565

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !568, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11uniform2StrIiEEDav(%class.ArgWriter* %14), !dbg !569
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !570
  store i1 true, i1* %4, align 1, !dbg !571
  br label %17, !dbg !571

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !572
  br label %17, !dbg !572

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !573
  ret i1 %18, !dbg !573
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20VarArg2StrIfSuitableIi9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !574 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !575, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !575
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIiEENS_8EncodingEv(), !dbg !576
  %11 = sext i8 %10 to i32, !dbg !576
  %12 = icmp eq i32 %9, %11, !dbg !577
  br i1 %12, label %13, label %16, !dbg !575

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !578, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11varying2StrIiEEDav(%class.ArgWriter* %14), !dbg !579
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !580
  store i1 true, i1* %4, align 1, !dbg !581
  br label %17, !dbg !581

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !582
  br label %17, !dbg !582

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !583
  ret i1 %18, !dbg !583
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4UniformIiEENS_8EncodingEv() #25 comdat !dbg !584 {
  ret i8 2, !dbg !585
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11uniform2StrIiEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !586 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i8*, align 4
  %4 = alloca <100 x i8>, align 128
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i8* @_Z16cmType2SpecifierIiEPKcv(), !dbg !587
  store i8* %7, i8** %3, align 4, !dbg !588, !tbaa !17
  %8 = load i8*, i8** %3, align 4, !dbg !589, !tbaa !17
  %9 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %8, <100 x i8>* %4, i32 0, i32 99), !dbg !590
  store i32 %9, i32* %5, align 4, !dbg !591, !tbaa !13
  %10 = load i32, i32* %5, align 4, !dbg !592, !tbaa !13
  %11 = trunc i32 %10 to i16, !dbg !592
  %12 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !593, !tbaa !7
  %13 = mul i16 %11, 1, !dbg !593
  %14 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %12, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %13, i32 undef, i1 true), !dbg !593
  store <100 x i8> %14, <100 x i8>* %4, align 128, !dbg !593, !tbaa !7
  call void @_ZN9ArgWriter8WriteArgIiEEvv(%class.ArgWriter* %6), !dbg !594
  %15 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !595, !tbaa !7
  ret <100 x i8> %15, !dbg !596
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_Z16cmType2SpecifierIiEPKcv() #25 comdat !dbg !597 {
  %1 = call i8* @_ZN9PrintInfo14type2SpecifierIiEEPKcv(), !dbg !598
  ret i8* %1, !dbg !599
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriter8WriteArgIiEEvv(%class.ArgWriter*) #25 comdat align 2 !dbg !600 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !601
  store i32 %7, i32* %3, align 4, !dbg !602, !tbaa !13
  store i32 0, i32* %4, align 4, !dbg !603, !tbaa !13
  %8 = call i32 @llvm.genx.predefined.surface(i32 2), !dbg !604
  store i32 %8, i32* %5, align 4, !dbg !605, !tbaa !31
  %9 = load i32, i32* %5, align 4, !dbg !606, !tbaa !31
  %10 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !607
  %11 = load i32, i32* %10, align 8, !dbg !607, !tbaa !59
  %12 = load i32, i32* %3, align 4, !dbg !608, !tbaa !13
  %13 = load i32, i32* %4, align 4, !dbg !609, !tbaa !13
  call void @_ZN7details18_cm_print_args_rawIiEEv15cm_surfaceindexjjj(i32 %9, i32 %11, i32 %12, i32 %13), !dbg !610
  %14 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !611
  %15 = load i32, i32* %14, align 8, !dbg !612, !tbaa !59
  %16 = add i32 %15, 32, !dbg !612
  store i32 %16, i32* %14, align 8, !dbg !612, !tbaa !59
  ret void, !dbg !613
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_ZN9PrintInfo14type2SpecifierIiEEPKcv() #25 comdat !dbg !614 {
  ret i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.5, i32 0, i32 0), !dbg !615
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details18_cm_print_args_rawIiEEv15cm_surfaceindexjjj(i32, i32, i32, i32) #10 comdat {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca <8 x i32>, align 32
  store i32 %0, i32* %5, align 4, !tbaa !31
  store i32 %1, i32* %6, align 4, !tbaa !13
  store i32 %2, i32* %7, align 4, !tbaa !13
  store i32 %3, i32* %8, align 4, !tbaa !13
  store <8 x i32> zeroinitializer, <8 x i32>* %9, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %11 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %10, <1 x i32> <i32 3>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %11, <8 x i32>* %9, align 32, !tbaa !7
  %12 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %13 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %12, <1 x i32> <i32 3>, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true)
  store <8 x i32> %13, <8 x i32>* %9, align 32, !tbaa !7
  %14 = load i32, i32* %7, align 4, !tbaa !13
  %15 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %16 = insertelement <1 x i32> undef, i32 %14, i32 0
  %17 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %15, <1 x i32> %16, i32 0, i32 1, i32 0, i16 24, i32 undef, i1 true)
  store <8 x i32> %17, <8 x i32>* %9, align 32, !tbaa !7
  %18 = load i32, i32* %5, align 4, !tbaa !31
  %19 = load i32, i32* %6, align 4, !tbaa !13
  %20 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  call void @_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_(i32 %18, i32 %19, <8 x i32> %20)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_(i32, i32, <8 x i32>) #10 comdat {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca <8 x i32>, align 32
  %7 = alloca i32, align 4
  store i32 %0, i32* %4, align 4, !tbaa !31
  store i32 %1, i32* %5, align 4, !tbaa !13
  store <8 x i32> %2, <8 x i32>* %6, align 32, !tbaa !7
  store i32 32, i32* %7, align 4, !tbaa !13
  %8 = load i32, i32* %4, align 4, !tbaa !31
  %9 = load i32, i32* %5, align 4, !tbaa !13
  %10 = load <8 x i32>, <8 x i32>* %6, align 32, !tbaa !7
  %11 = udiv exact i32 %9, 16
  call void @llvm.genx.oword.st.v8i32(i32 %8, i32 %11, <8 x i32> %10)
  ret void
}

declare dso_local void @_ZN7details31__cm_intrinsic_impl_oword_writeIjLi8EEEv15cm_surfaceindexiu2CMvbT0__T_(i32, i32, <8 x i32>) #24

; Function Attrs: nounwind
declare void @llvm.genx.oword.st.v8i32(i32, i32, <8 x i32>) #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4VaryingIiEENS_8EncodingEv() #25 comdat !dbg !617 {
  ret i8 10, !dbg !618
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11varying2StrIiEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !619 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !620, !tbaa !7
  %8 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %7, <1 x i8> <i8 91>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !620
  store <100 x i8> %8, <100 x i8>* %3, align 128, !dbg !620, !tbaa !7
  store i32 1, i32* %4, align 4, !dbg !621, !tbaa !13
  store i32 0, i32* %5, align 4, !dbg !622, !tbaa !13
  br label %9, !dbg !623

; <label>:9:                                      ; preds = %58, %1
  %10 = load i32, i32* %5, align 4, !dbg !624, !tbaa !13
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !625
  %12 = load i32, i32* %11, align 4, !dbg !625, !tbaa !68
  %13 = icmp slt i32 %10, %12, !dbg !626
  br i1 %13, label %14, label %61, !dbg !627

; <label>:14:                                     ; preds = %9
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 4, !dbg !628
  %16 = load i64, i64* %15, align 8, !dbg !628, !tbaa !71
  %17 = load i32, i32* %5, align 4, !dbg !629, !tbaa !13
  %18 = zext i32 %17 to i64, !dbg !630
  %19 = shl i64 1, %18, !dbg !630
  %20 = and i64 %16, %19, !dbg !631
  %21 = icmp ne i64 %20, 0, !dbg !628
  br i1 %21, label %22, label %32, !dbg !628

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %4, align 4, !dbg !632, !tbaa !13
  %24 = call i32 @_Z21requiredSpace4VecElemIiL9LaneState1EEiv(), !dbg !633
  %25 = sub i32 100, %24, !dbg !634
  %26 = sub i32 %25, 1, !dbg !635
  %27 = icmp uge i32 %23, %26, !dbg !636
  br i1 %27, label %28, label %29, !dbg !632

; <label>:28:                                     ; preds = %22
  br label %61, !dbg !637

; <label>:29:                                     ; preds = %22
  %30 = load i32, i32* %4, align 4, !dbg !638, !tbaa !13
  %31 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIiL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %30), !dbg !639
  store i32 %31, i32* %4, align 4, !dbg !640, !tbaa !13
  br label %42, !dbg !641

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %4, align 4, !dbg !642, !tbaa !13
  %34 = call i32 @_Z21requiredSpace4VecElemIiL9LaneState0EEiv(), !dbg !643
  %35 = sub i32 100, %34, !dbg !644
  %36 = sub i32 %35, 1, !dbg !645
  %37 = icmp uge i32 %33, %36, !dbg !646
  br i1 %37, label %38, label %39, !dbg !642

; <label>:38:                                     ; preds = %32
  br label %61, !dbg !647

; <label>:39:                                     ; preds = %32
  %40 = load i32, i32* %4, align 4, !dbg !648, !tbaa !13
  %41 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIiL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %40), !dbg !649
  store i32 %41, i32* %4, align 4, !dbg !650, !tbaa !13
  br label %42

; <label>:42:                                     ; preds = %39, %29
  %43 = load i32, i32* %5, align 4, !dbg !651, !tbaa !13
  %44 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !652
  %45 = load i32, i32* %44, align 4, !dbg !652, !tbaa !68
  %46 = sub nsw i32 %45, 1, !dbg !653
  %47 = icmp eq i32 %43, %46, !dbg !654
  %48 = zext i1 %47 to i64, !dbg !651
  %49 = select i1 %47, i8 93, i8 44, !dbg !651
  %50 = load i32, i32* %4, align 4, !dbg !655, !tbaa !13
  %51 = trunc i32 %50 to i16, !dbg !655
  %52 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !656, !tbaa !7
  %53 = insertelement <1 x i8> undef, i8 %49, i32 0, !dbg !656
  %54 = mul i16 %51, 1, !dbg !656
  %55 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %52, <1 x i8> %53, i32 0, i32 1, i32 0, i16 %54, i32 undef, i1 true), !dbg !656
  store <100 x i8> %55, <100 x i8>* %3, align 128, !dbg !656, !tbaa !7
  %56 = load i32, i32* %4, align 4, !dbg !657, !tbaa !13
  %57 = add nsw i32 %56, 1, !dbg !657
  store i32 %57, i32* %4, align 4, !dbg !657, !tbaa !13
  call void @_ZN9ArgWriter8WriteArgIiEEvv(%class.ArgWriter* %6), !dbg !658
  br label %58, !dbg !659

; <label>:58:                                     ; preds = %42
  %59 = load i32, i32* %5, align 4, !dbg !660, !tbaa !13
  %60 = add nsw i32 %59, 1, !dbg !660
  store i32 %60, i32* %5, align 4, !dbg !660, !tbaa !13
  br label %9, !dbg !627, !llvm.loop !661

; <label>:61:                                     ; preds = %38, %28, %9
  %62 = load i32, i32* %4, align 4, !dbg !662, !tbaa !13
  %63 = trunc i32 %62 to i16, !dbg !662
  %64 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !663, !tbaa !7
  %65 = mul i16 %63, 1, !dbg !663
  %66 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %64, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %65, i32 undef, i1 true), !dbg !663
  store <100 x i8> %66, <100 x i8>* %3, align 128, !dbg !663, !tbaa !7
  %67 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !664, !tbaa !7
  ret <100 x i8> %67, !dbg !665
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIiL9LaneState1EEiv() #25 comdat !dbg !666 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIiEPKcv(), !dbg !667
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !668
  store i32 %3, i32* %1, align 4, !dbg !669, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !670, !tbaa !13
  %5 = add nsw i32 %4, 1, !dbg !671
  ret i32 %5, !dbg !672
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIiL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !673 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIiEPKcv(), !dbg !674
  store i8* %9, i8** %7, align 4, !dbg !675, !tbaa !17
  %10 = load i8*, i8** %7, align 4, !dbg !676, !tbaa !17
  %11 = load i32, i32* %6, align 4, !dbg !677, !tbaa !13
  %12 = load i32, i32* %6, align 4, !dbg !678, !tbaa !13
  %13 = sub i32 100, %12, !dbg !679
  %14 = sub i32 %13, 1, !dbg !680
  %15 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %10, <100 x i8>* %1, i32 %11, i32 %14), !dbg !681
  %16 = load i32, i32* %6, align 4, !dbg !682, !tbaa !13
  %17 = add nsw i32 %16, %15, !dbg !682
  store i32 %17, i32* %6, align 4, !dbg !682, !tbaa !13
  %18 = load i32, i32* %6, align 4, !dbg !683, !tbaa !13
  ret i32 %18, !dbg !684
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIiL9LaneState0EEiv() #25 comdat !dbg !685 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIiEPKcv(), !dbg !686
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !687
  store i32 %3, i32* %1, align 4, !dbg !688, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !689, !tbaa !13
  %5 = add nsw i32 %4, 5, !dbg !690
  ret i32 %5, !dbg !691
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIiL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !692 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIiEPKcv(), !dbg !693
  store i8* %9, i8** %7, align 4, !dbg !694, !tbaa !17
  %10 = load i32, i32* %6, align 4, !dbg !695, !tbaa !13
  %11 = load i32, i32* %6, align 4, !dbg !696, !tbaa !13
  %12 = sub i32 100, %11, !dbg !697
  %13 = sub i32 %12, 1, !dbg !698
  %14 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), <100 x i8>* %1, i32 %10, i32 %13), !dbg !699
  %15 = load i32, i32* %6, align 4, !dbg !700, !tbaa !13
  %16 = add nsw i32 %15, %14, !dbg !700
  store i32 %16, i32* %6, align 4, !dbg !700, !tbaa !13
  %17 = load i8*, i8** %7, align 4, !dbg !701, !tbaa !17
  %18 = load i32, i32* %6, align 4, !dbg !702, !tbaa !13
  %19 = load i32, i32* %6, align 4, !dbg !703, !tbaa !13
  %20 = sub i32 100, %19, !dbg !704
  %21 = sub i32 %20, 1, !dbg !705
  %22 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %17, <100 x i8>* %1, i32 %18, i32 %21), !dbg !706
  %23 = load i32, i32* %6, align 4, !dbg !707, !tbaa !13
  %24 = add nsw i32 %23, %22, !dbg !707
  store i32 %24, i32* %6, align 4, !dbg !707, !tbaa !13
  %25 = load i32, i32* %6, align 4, !dbg !708, !tbaa !13
  %26 = load i32, i32* %6, align 4, !dbg !709, !tbaa !13
  %27 = sub i32 100, %26, !dbg !710
  %28 = sub i32 %27, 1, !dbg !711
  %29 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0), <100 x i8>* %1, i32 %25, i32 %28), !dbg !712
  %30 = load i32, i32* %6, align 4, !dbg !713, !tbaa !13
  %31 = add nsw i32 %30, %29, !dbg !713
  store i32 %31, i32* %6, align 4, !dbg !713, !tbaa !13
  %32 = load i32, i32* %6, align 4, !dbg !714, !tbaa !13
  ret i32 %32, !dbg !715
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20UniArg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !716 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !717, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !717
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIjEENS_8EncodingEv(), !dbg !718
  %11 = sext i8 %10 to i32, !dbg !718
  %12 = icmp eq i32 %9, %11, !dbg !719
  br i1 %12, label %13, label %16, !dbg !717

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !720, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11uniform2StrIjEEDav(%class.ArgWriter* %14), !dbg !721
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !722
  store i1 true, i1* %4, align 1, !dbg !723
  br label %17, !dbg !723

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !724
  br label %17, !dbg !724

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !725
  ret i1 %18, !dbg !725
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20VarArg2StrIfSuitableIj9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !726 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !727, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !727
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIjEENS_8EncodingEv(), !dbg !728
  %11 = sext i8 %10 to i32, !dbg !728
  %12 = icmp eq i32 %9, %11, !dbg !729
  br i1 %12, label %13, label %16, !dbg !727

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !730, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11varying2StrIjEEDav(%class.ArgWriter* %14), !dbg !731
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !732
  store i1 true, i1* %4, align 1, !dbg !733
  br label %17, !dbg !733

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !734
  br label %17, !dbg !734

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !735
  ret i1 %18, !dbg !735
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4UniformIjEENS_8EncodingEv() #25 comdat !dbg !736 {
  ret i8 3, !dbg !737
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11uniform2StrIjEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !738 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i8*, align 4
  %4 = alloca <100 x i8>, align 128
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i8* @_Z16cmType2SpecifierIjEPKcv(), !dbg !739
  store i8* %7, i8** %3, align 4, !dbg !740, !tbaa !17
  %8 = load i8*, i8** %3, align 4, !dbg !741, !tbaa !17
  %9 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %8, <100 x i8>* %4, i32 0, i32 99), !dbg !742
  store i32 %9, i32* %5, align 4, !dbg !743, !tbaa !13
  %10 = load i32, i32* %5, align 4, !dbg !744, !tbaa !13
  %11 = trunc i32 %10 to i16, !dbg !744
  %12 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !745, !tbaa !7
  %13 = mul i16 %11, 1, !dbg !745
  %14 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %12, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %13, i32 undef, i1 true), !dbg !745
  store <100 x i8> %14, <100 x i8>* %4, align 128, !dbg !745, !tbaa !7
  call void @_ZN9ArgWriter8WriteArgIjEEvv(%class.ArgWriter* %6), !dbg !746
  %15 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !747, !tbaa !7
  ret <100 x i8> %15, !dbg !748
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_Z16cmType2SpecifierIjEPKcv() #25 comdat !dbg !749 {
  %1 = call i8* @_ZN9PrintInfo14type2SpecifierIjEEPKcv(), !dbg !750
  ret i8* %1, !dbg !751
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriter8WriteArgIjEEvv(%class.ArgWriter*) #25 comdat align 2 !dbg !752 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !753
  store i32 %7, i32* %3, align 4, !dbg !754, !tbaa !13
  store i32 0, i32* %4, align 4, !dbg !755, !tbaa !13
  %8 = call i32 @llvm.genx.predefined.surface(i32 2), !dbg !756
  store i32 %8, i32* %5, align 4, !dbg !757, !tbaa !31
  %9 = load i32, i32* %5, align 4, !dbg !758, !tbaa !31
  %10 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !759
  %11 = load i32, i32* %10, align 8, !dbg !759, !tbaa !59
  %12 = load i32, i32* %3, align 4, !dbg !760, !tbaa !13
  %13 = load i32, i32* %4, align 4, !dbg !761, !tbaa !13
  call void @_ZN7details18_cm_print_args_rawIjEEv15cm_surfaceindexjjj(i32 %9, i32 %11, i32 %12, i32 %13), !dbg !762
  %14 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !763
  %15 = load i32, i32* %14, align 8, !dbg !764, !tbaa !59
  %16 = add i32 %15, 32, !dbg !764
  store i32 %16, i32* %14, align 8, !dbg !764, !tbaa !59
  ret void, !dbg !765
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_ZN9PrintInfo14type2SpecifierIjEEPKcv() #25 comdat !dbg !766 {
  ret i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.6, i32 0, i32 0), !dbg !767
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details18_cm_print_args_rawIjEEv15cm_surfaceindexjjj(i32, i32, i32, i32) #10 comdat {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca <8 x i32>, align 32
  store i32 %0, i32* %5, align 4, !tbaa !31
  store i32 %1, i32* %6, align 4, !tbaa !13
  store i32 %2, i32* %7, align 4, !tbaa !13
  store i32 %3, i32* %8, align 4, !tbaa !13
  store <8 x i32> zeroinitializer, <8 x i32>* %9, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %11 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %10, <1 x i32> <i32 3>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %11, <8 x i32>* %9, align 32, !tbaa !7
  %12 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %13 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %12, <1 x i32> <i32 4>, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true)
  store <8 x i32> %13, <8 x i32>* %9, align 32, !tbaa !7
  %14 = load i32, i32* %7, align 4, !tbaa !13
  %15 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %16 = insertelement <1 x i32> undef, i32 %14, i32 0
  %17 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %15, <1 x i32> %16, i32 0, i32 1, i32 0, i16 24, i32 undef, i1 true)
  store <8 x i32> %17, <8 x i32>* %9, align 32, !tbaa !7
  %18 = load i32, i32* %5, align 4, !tbaa !31
  %19 = load i32, i32* %6, align 4, !tbaa !13
  %20 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  call void @_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_(i32 %18, i32 %19, <8 x i32> %20)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4VaryingIjEENS_8EncodingEv() #25 comdat !dbg !768 {
  ret i8 11, !dbg !769
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11varying2StrIjEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !770 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !771, !tbaa !7
  %8 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %7, <1 x i8> <i8 91>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !771
  store <100 x i8> %8, <100 x i8>* %3, align 128, !dbg !771, !tbaa !7
  store i32 1, i32* %4, align 4, !dbg !772, !tbaa !13
  store i32 0, i32* %5, align 4, !dbg !773, !tbaa !13
  br label %9, !dbg !774

; <label>:9:                                      ; preds = %58, %1
  %10 = load i32, i32* %5, align 4, !dbg !775, !tbaa !13
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !776
  %12 = load i32, i32* %11, align 4, !dbg !776, !tbaa !68
  %13 = icmp slt i32 %10, %12, !dbg !777
  br i1 %13, label %14, label %61, !dbg !778

; <label>:14:                                     ; preds = %9
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 4, !dbg !779
  %16 = load i64, i64* %15, align 8, !dbg !779, !tbaa !71
  %17 = load i32, i32* %5, align 4, !dbg !780, !tbaa !13
  %18 = zext i32 %17 to i64, !dbg !781
  %19 = shl i64 1, %18, !dbg !781
  %20 = and i64 %16, %19, !dbg !782
  %21 = icmp ne i64 %20, 0, !dbg !779
  br i1 %21, label %22, label %32, !dbg !779

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %4, align 4, !dbg !783, !tbaa !13
  %24 = call i32 @_Z21requiredSpace4VecElemIjL9LaneState1EEiv(), !dbg !784
  %25 = sub i32 100, %24, !dbg !785
  %26 = sub i32 %25, 1, !dbg !786
  %27 = icmp uge i32 %23, %26, !dbg !787
  br i1 %27, label %28, label %29, !dbg !783

; <label>:28:                                     ; preds = %22
  br label %61, !dbg !788

; <label>:29:                                     ; preds = %22
  %30 = load i32, i32* %4, align 4, !dbg !789, !tbaa !13
  %31 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIjL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %30), !dbg !790
  store i32 %31, i32* %4, align 4, !dbg !791, !tbaa !13
  br label %42, !dbg !792

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %4, align 4, !dbg !793, !tbaa !13
  %34 = call i32 @_Z21requiredSpace4VecElemIjL9LaneState0EEiv(), !dbg !794
  %35 = sub i32 100, %34, !dbg !795
  %36 = sub i32 %35, 1, !dbg !796
  %37 = icmp uge i32 %33, %36, !dbg !797
  br i1 %37, label %38, label %39, !dbg !793

; <label>:38:                                     ; preds = %32
  br label %61, !dbg !798

; <label>:39:                                     ; preds = %32
  %40 = load i32, i32* %4, align 4, !dbg !799, !tbaa !13
  %41 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIjL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %40), !dbg !800
  store i32 %41, i32* %4, align 4, !dbg !801, !tbaa !13
  br label %42

; <label>:42:                                     ; preds = %39, %29
  %43 = load i32, i32* %5, align 4, !dbg !802, !tbaa !13
  %44 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !803
  %45 = load i32, i32* %44, align 4, !dbg !803, !tbaa !68
  %46 = sub nsw i32 %45, 1, !dbg !804
  %47 = icmp eq i32 %43, %46, !dbg !805
  %48 = zext i1 %47 to i64, !dbg !802
  %49 = select i1 %47, i8 93, i8 44, !dbg !802
  %50 = load i32, i32* %4, align 4, !dbg !806, !tbaa !13
  %51 = trunc i32 %50 to i16, !dbg !806
  %52 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !807, !tbaa !7
  %53 = insertelement <1 x i8> undef, i8 %49, i32 0, !dbg !807
  %54 = mul i16 %51, 1, !dbg !807
  %55 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %52, <1 x i8> %53, i32 0, i32 1, i32 0, i16 %54, i32 undef, i1 true), !dbg !807
  store <100 x i8> %55, <100 x i8>* %3, align 128, !dbg !807, !tbaa !7
  %56 = load i32, i32* %4, align 4, !dbg !808, !tbaa !13
  %57 = add nsw i32 %56, 1, !dbg !808
  store i32 %57, i32* %4, align 4, !dbg !808, !tbaa !13
  call void @_ZN9ArgWriter8WriteArgIjEEvv(%class.ArgWriter* %6), !dbg !809
  br label %58, !dbg !810

; <label>:58:                                     ; preds = %42
  %59 = load i32, i32* %5, align 4, !dbg !811, !tbaa !13
  %60 = add nsw i32 %59, 1, !dbg !811
  store i32 %60, i32* %5, align 4, !dbg !811, !tbaa !13
  br label %9, !dbg !778, !llvm.loop !812

; <label>:61:                                     ; preds = %38, %28, %9
  %62 = load i32, i32* %4, align 4, !dbg !813, !tbaa !13
  %63 = trunc i32 %62 to i16, !dbg !813
  %64 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !814, !tbaa !7
  %65 = mul i16 %63, 1, !dbg !814
  %66 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %64, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %65, i32 undef, i1 true), !dbg !814
  store <100 x i8> %66, <100 x i8>* %3, align 128, !dbg !814, !tbaa !7
  %67 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !815, !tbaa !7
  ret <100 x i8> %67, !dbg !816
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIjL9LaneState1EEiv() #25 comdat !dbg !817 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIjEPKcv(), !dbg !818
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !819
  store i32 %3, i32* %1, align 4, !dbg !820, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !821, !tbaa !13
  %5 = add nsw i32 %4, 1, !dbg !822
  ret i32 %5, !dbg !823
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIjL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !824 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIjEPKcv(), !dbg !825
  store i8* %9, i8** %7, align 4, !dbg !826, !tbaa !17
  %10 = load i8*, i8** %7, align 4, !dbg !827, !tbaa !17
  %11 = load i32, i32* %6, align 4, !dbg !828, !tbaa !13
  %12 = load i32, i32* %6, align 4, !dbg !829, !tbaa !13
  %13 = sub i32 100, %12, !dbg !830
  %14 = sub i32 %13, 1, !dbg !831
  %15 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %10, <100 x i8>* %1, i32 %11, i32 %14), !dbg !832
  %16 = load i32, i32* %6, align 4, !dbg !833, !tbaa !13
  %17 = add nsw i32 %16, %15, !dbg !833
  store i32 %17, i32* %6, align 4, !dbg !833, !tbaa !13
  %18 = load i32, i32* %6, align 4, !dbg !834, !tbaa !13
  ret i32 %18, !dbg !835
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIjL9LaneState0EEiv() #25 comdat !dbg !836 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIjEPKcv(), !dbg !837
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !838
  store i32 %3, i32* %1, align 4, !dbg !839, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !840, !tbaa !13
  %5 = add nsw i32 %4, 5, !dbg !841
  ret i32 %5, !dbg !842
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIjL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !843 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIjEPKcv(), !dbg !844
  store i8* %9, i8** %7, align 4, !dbg !845, !tbaa !17
  %10 = load i32, i32* %6, align 4, !dbg !846, !tbaa !13
  %11 = load i32, i32* %6, align 4, !dbg !847, !tbaa !13
  %12 = sub i32 100, %11, !dbg !848
  %13 = sub i32 %12, 1, !dbg !849
  %14 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), <100 x i8>* %1, i32 %10, i32 %13), !dbg !850
  %15 = load i32, i32* %6, align 4, !dbg !851, !tbaa !13
  %16 = add nsw i32 %15, %14, !dbg !851
  store i32 %16, i32* %6, align 4, !dbg !851, !tbaa !13
  %17 = load i8*, i8** %7, align 4, !dbg !852, !tbaa !17
  %18 = load i32, i32* %6, align 4, !dbg !853, !tbaa !13
  %19 = load i32, i32* %6, align 4, !dbg !854, !tbaa !13
  %20 = sub i32 100, %19, !dbg !855
  %21 = sub i32 %20, 1, !dbg !856
  %22 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %17, <100 x i8>* %1, i32 %18, i32 %21), !dbg !857
  %23 = load i32, i32* %6, align 4, !dbg !858, !tbaa !13
  %24 = add nsw i32 %23, %22, !dbg !858
  store i32 %24, i32* %6, align 4, !dbg !858, !tbaa !13
  %25 = load i32, i32* %6, align 4, !dbg !859, !tbaa !13
  %26 = load i32, i32* %6, align 4, !dbg !860, !tbaa !13
  %27 = sub i32 100, %26, !dbg !861
  %28 = sub i32 %27, 1, !dbg !862
  %29 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0), <100 x i8>* %1, i32 %25, i32 %28), !dbg !863
  %30 = load i32, i32* %6, align 4, !dbg !864, !tbaa !13
  %31 = add nsw i32 %30, %29, !dbg !864
  store i32 %31, i32* %6, align 4, !dbg !864, !tbaa !13
  %32 = load i32, i32* %6, align 4, !dbg !865, !tbaa !13
  ret i32 %32, !dbg !866
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20UniArg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !867 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !868, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !868
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIfEENS_8EncodingEv(), !dbg !869
  %11 = sext i8 %10 to i32, !dbg !869
  %12 = icmp eq i32 %9, %11, !dbg !870
  br i1 %12, label %13, label %16, !dbg !868

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !871, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11uniform2StrIfEEDav(%class.ArgWriter* %14), !dbg !872
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !873
  store i1 true, i1* %4, align 1, !dbg !874
  br label %17, !dbg !874

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !875
  br label %17, !dbg !875

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !876
  ret i1 %18, !dbg !876
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20VarArg2StrIfSuitableIf9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !877 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !878, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !878
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIfEENS_8EncodingEv(), !dbg !879
  %11 = sext i8 %10 to i32, !dbg !879
  %12 = icmp eq i32 %9, %11, !dbg !880
  br i1 %12, label %13, label %16, !dbg !878

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !881, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11varying2StrIfEEDav(%class.ArgWriter* %14), !dbg !882
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !883
  store i1 true, i1* %4, align 1, !dbg !884
  br label %17, !dbg !884

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !885
  br label %17, !dbg !885

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !886
  ret i1 %18, !dbg !886
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4UniformIfEENS_8EncodingEv() #25 comdat !dbg !887 {
  ret i8 4, !dbg !888
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11uniform2StrIfEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !889 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i8*, align 4
  %4 = alloca <100 x i8>, align 128
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i8* @_Z16cmType2SpecifierIfEPKcv(), !dbg !890
  store i8* %7, i8** %3, align 4, !dbg !891, !tbaa !17
  %8 = load i8*, i8** %3, align 4, !dbg !892, !tbaa !17
  %9 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %8, <100 x i8>* %4, i32 0, i32 99), !dbg !893
  store i32 %9, i32* %5, align 4, !dbg !894, !tbaa !13
  %10 = load i32, i32* %5, align 4, !dbg !895, !tbaa !13
  %11 = trunc i32 %10 to i16, !dbg !895
  %12 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !896, !tbaa !7
  %13 = mul i16 %11, 1, !dbg !896
  %14 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %12, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %13, i32 undef, i1 true), !dbg !896
  store <100 x i8> %14, <100 x i8>* %4, align 128, !dbg !896, !tbaa !7
  call void @_ZN9ArgWriter8WriteArgIfEEvv(%class.ArgWriter* %6), !dbg !897
  %15 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !898, !tbaa !7
  ret <100 x i8> %15, !dbg !899
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_Z16cmType2SpecifierIfEPKcv() #25 comdat !dbg !900 {
  %1 = call i8* @_ZN9PrintInfo14type2SpecifierIfEEPKcv(), !dbg !901
  ret i8* %1, !dbg !902
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriter8WriteArgIfEEvv(%class.ArgWriter*) #25 comdat align 2 !dbg !903 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !904
  store i32 %7, i32* %3, align 4, !dbg !905, !tbaa !13
  store i32 0, i32* %4, align 4, !dbg !906, !tbaa !13
  %8 = call i32 @llvm.genx.predefined.surface(i32 2), !dbg !907
  store i32 %8, i32* %5, align 4, !dbg !908, !tbaa !31
  %9 = load i32, i32* %5, align 4, !dbg !909, !tbaa !31
  %10 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !910
  %11 = load i32, i32* %10, align 8, !dbg !910, !tbaa !59
  %12 = load i32, i32* %3, align 4, !dbg !911, !tbaa !13
  %13 = load i32, i32* %4, align 4, !dbg !912, !tbaa !13
  call void @_ZN7details18_cm_print_args_rawIfEEv15cm_surfaceindexjjj(i32 %9, i32 %11, i32 %12, i32 %13), !dbg !913
  %14 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !914
  %15 = load i32, i32* %14, align 8, !dbg !915, !tbaa !59
  %16 = add i32 %15, 32, !dbg !915
  store i32 %16, i32* %14, align 8, !dbg !915, !tbaa !59
  ret void, !dbg !916
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_ZN9PrintInfo14type2SpecifierIfEEPKcv() #25 comdat !dbg !917 {
  ret i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.7, i32 0, i32 0), !dbg !918
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details18_cm_print_args_rawIfEEv15cm_surfaceindexjjj(i32, i32, i32, i32) #10 comdat {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca <8 x i32>, align 32
  store i32 %0, i32* %5, align 4, !tbaa !31
  store i32 %1, i32* %6, align 4, !tbaa !13
  store i32 %2, i32* %7, align 4, !tbaa !13
  store i32 %3, i32* %8, align 4, !tbaa !13
  store <8 x i32> zeroinitializer, <8 x i32>* %9, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %11 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %10, <1 x i32> <i32 3>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %11, <8 x i32>* %9, align 32, !tbaa !7
  %12 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %13 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %12, <1 x i32> <i32 2>, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true)
  store <8 x i32> %13, <8 x i32>* %9, align 32, !tbaa !7
  %14 = load i32, i32* %7, align 4, !tbaa !13
  %15 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %16 = insertelement <1 x i32> undef, i32 %14, i32 0
  %17 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %15, <1 x i32> %16, i32 0, i32 1, i32 0, i16 24, i32 undef, i1 true)
  store <8 x i32> %17, <8 x i32>* %9, align 32, !tbaa !7
  %18 = load i32, i32* %5, align 4, !tbaa !31
  %19 = load i32, i32* %6, align 4, !tbaa !13
  %20 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  call void @_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_(i32 %18, i32 %19, <8 x i32> %20)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4VaryingIfEENS_8EncodingEv() #25 comdat !dbg !919 {
  ret i8 12, !dbg !920
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11varying2StrIfEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !921 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !922, !tbaa !7
  %8 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %7, <1 x i8> <i8 91>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !922
  store <100 x i8> %8, <100 x i8>* %3, align 128, !dbg !922, !tbaa !7
  store i32 1, i32* %4, align 4, !dbg !923, !tbaa !13
  store i32 0, i32* %5, align 4, !dbg !924, !tbaa !13
  br label %9, !dbg !925

; <label>:9:                                      ; preds = %58, %1
  %10 = load i32, i32* %5, align 4, !dbg !926, !tbaa !13
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !927
  %12 = load i32, i32* %11, align 4, !dbg !927, !tbaa !68
  %13 = icmp slt i32 %10, %12, !dbg !928
  br i1 %13, label %14, label %61, !dbg !929

; <label>:14:                                     ; preds = %9
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 4, !dbg !930
  %16 = load i64, i64* %15, align 8, !dbg !930, !tbaa !71
  %17 = load i32, i32* %5, align 4, !dbg !931, !tbaa !13
  %18 = zext i32 %17 to i64, !dbg !932
  %19 = shl i64 1, %18, !dbg !932
  %20 = and i64 %16, %19, !dbg !933
  %21 = icmp ne i64 %20, 0, !dbg !930
  br i1 %21, label %22, label %32, !dbg !930

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %4, align 4, !dbg !934, !tbaa !13
  %24 = call i32 @_Z21requiredSpace4VecElemIfL9LaneState1EEiv(), !dbg !935
  %25 = sub i32 100, %24, !dbg !936
  %26 = sub i32 %25, 1, !dbg !937
  %27 = icmp uge i32 %23, %26, !dbg !938
  br i1 %27, label %28, label %29, !dbg !934

; <label>:28:                                     ; preds = %22
  br label %61, !dbg !939

; <label>:29:                                     ; preds = %22
  %30 = load i32, i32* %4, align 4, !dbg !940, !tbaa !13
  %31 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIfL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %30), !dbg !941
  store i32 %31, i32* %4, align 4, !dbg !942, !tbaa !13
  br label %42, !dbg !943

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %4, align 4, !dbg !944, !tbaa !13
  %34 = call i32 @_Z21requiredSpace4VecElemIfL9LaneState0EEiv(), !dbg !945
  %35 = sub i32 100, %34, !dbg !946
  %36 = sub i32 %35, 1, !dbg !947
  %37 = icmp uge i32 %33, %36, !dbg !948
  br i1 %37, label %38, label %39, !dbg !944

; <label>:38:                                     ; preds = %32
  br label %61, !dbg !949

; <label>:39:                                     ; preds = %32
  %40 = load i32, i32* %4, align 4, !dbg !950, !tbaa !13
  %41 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIfL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %40), !dbg !951
  store i32 %41, i32* %4, align 4, !dbg !952, !tbaa !13
  br label %42

; <label>:42:                                     ; preds = %39, %29
  %43 = load i32, i32* %5, align 4, !dbg !953, !tbaa !13
  %44 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !954
  %45 = load i32, i32* %44, align 4, !dbg !954, !tbaa !68
  %46 = sub nsw i32 %45, 1, !dbg !955
  %47 = icmp eq i32 %43, %46, !dbg !956
  %48 = zext i1 %47 to i64, !dbg !953
  %49 = select i1 %47, i8 93, i8 44, !dbg !953
  %50 = load i32, i32* %4, align 4, !dbg !957, !tbaa !13
  %51 = trunc i32 %50 to i16, !dbg !957
  %52 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !958, !tbaa !7
  %53 = insertelement <1 x i8> undef, i8 %49, i32 0, !dbg !958
  %54 = mul i16 %51, 1, !dbg !958
  %55 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %52, <1 x i8> %53, i32 0, i32 1, i32 0, i16 %54, i32 undef, i1 true), !dbg !958
  store <100 x i8> %55, <100 x i8>* %3, align 128, !dbg !958, !tbaa !7
  %56 = load i32, i32* %4, align 4, !dbg !959, !tbaa !13
  %57 = add nsw i32 %56, 1, !dbg !959
  store i32 %57, i32* %4, align 4, !dbg !959, !tbaa !13
  call void @_ZN9ArgWriter8WriteArgIfEEvv(%class.ArgWriter* %6), !dbg !960
  br label %58, !dbg !961

; <label>:58:                                     ; preds = %42
  %59 = load i32, i32* %5, align 4, !dbg !962, !tbaa !13
  %60 = add nsw i32 %59, 1, !dbg !962
  store i32 %60, i32* %5, align 4, !dbg !962, !tbaa !13
  br label %9, !dbg !929, !llvm.loop !963

; <label>:61:                                     ; preds = %38, %28, %9
  %62 = load i32, i32* %4, align 4, !dbg !964, !tbaa !13
  %63 = trunc i32 %62 to i16, !dbg !964
  %64 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !965, !tbaa !7
  %65 = mul i16 %63, 1, !dbg !965
  %66 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %64, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %65, i32 undef, i1 true), !dbg !965
  store <100 x i8> %66, <100 x i8>* %3, align 128, !dbg !965, !tbaa !7
  %67 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !966, !tbaa !7
  ret <100 x i8> %67, !dbg !967
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIfL9LaneState1EEiv() #25 comdat !dbg !968 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIfEPKcv(), !dbg !969
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !970
  store i32 %3, i32* %1, align 4, !dbg !971, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !972, !tbaa !13
  %5 = add nsw i32 %4, 1, !dbg !973
  ret i32 %5, !dbg !974
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIfL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !975 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIfEPKcv(), !dbg !976
  store i8* %9, i8** %7, align 4, !dbg !977, !tbaa !17
  %10 = load i8*, i8** %7, align 4, !dbg !978, !tbaa !17
  %11 = load i32, i32* %6, align 4, !dbg !979, !tbaa !13
  %12 = load i32, i32* %6, align 4, !dbg !980, !tbaa !13
  %13 = sub i32 100, %12, !dbg !981
  %14 = sub i32 %13, 1, !dbg !982
  %15 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %10, <100 x i8>* %1, i32 %11, i32 %14), !dbg !983
  %16 = load i32, i32* %6, align 4, !dbg !984, !tbaa !13
  %17 = add nsw i32 %16, %15, !dbg !984
  store i32 %17, i32* %6, align 4, !dbg !984, !tbaa !13
  %18 = load i32, i32* %6, align 4, !dbg !985, !tbaa !13
  ret i32 %18, !dbg !986
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIfL9LaneState0EEiv() #25 comdat !dbg !987 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIfEPKcv(), !dbg !988
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !989
  store i32 %3, i32* %1, align 4, !dbg !990, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !991, !tbaa !13
  %5 = add nsw i32 %4, 5, !dbg !992
  ret i32 %5, !dbg !993
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIfL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !994 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIfEPKcv(), !dbg !995
  store i8* %9, i8** %7, align 4, !dbg !996, !tbaa !17
  %10 = load i32, i32* %6, align 4, !dbg !997, !tbaa !13
  %11 = load i32, i32* %6, align 4, !dbg !998, !tbaa !13
  %12 = sub i32 100, %11, !dbg !999
  %13 = sub i32 %12, 1, !dbg !1000
  %14 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), <100 x i8>* %1, i32 %10, i32 %13), !dbg !1001
  %15 = load i32, i32* %6, align 4, !dbg !1002, !tbaa !13
  %16 = add nsw i32 %15, %14, !dbg !1002
  store i32 %16, i32* %6, align 4, !dbg !1002, !tbaa !13
  %17 = load i8*, i8** %7, align 4, !dbg !1003, !tbaa !17
  %18 = load i32, i32* %6, align 4, !dbg !1004, !tbaa !13
  %19 = load i32, i32* %6, align 4, !dbg !1005, !tbaa !13
  %20 = sub i32 100, %19, !dbg !1006
  %21 = sub i32 %20, 1, !dbg !1007
  %22 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %17, <100 x i8>* %1, i32 %18, i32 %21), !dbg !1008
  %23 = load i32, i32* %6, align 4, !dbg !1009, !tbaa !13
  %24 = add nsw i32 %23, %22, !dbg !1009
  store i32 %24, i32* %6, align 4, !dbg !1009, !tbaa !13
  %25 = load i32, i32* %6, align 4, !dbg !1010, !tbaa !13
  %26 = load i32, i32* %6, align 4, !dbg !1011, !tbaa !13
  %27 = sub i32 100, %26, !dbg !1012
  %28 = sub i32 %27, 1, !dbg !1013
  %29 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0), <100 x i8>* %1, i32 %25, i32 %28), !dbg !1014
  %30 = load i32, i32* %6, align 4, !dbg !1015, !tbaa !13
  %31 = add nsw i32 %30, %29, !dbg !1015
  store i32 %31, i32* %6, align 4, !dbg !1015, !tbaa !13
  %32 = load i32, i32* %6, align 4, !dbg !1016, !tbaa !13
  ret i32 %32, !dbg !1017
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20UniArg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !1018 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !1019, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !1019
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIxEENS_8EncodingEv(), !dbg !1020
  %11 = sext i8 %10 to i32, !dbg !1020
  %12 = icmp eq i32 %9, %11, !dbg !1021
  br i1 %12, label %13, label %16, !dbg !1019

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !1022, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11uniform2StrIxEEDav(%class.ArgWriter* %14), !dbg !1023
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !1024
  store i1 true, i1* %4, align 1, !dbg !1025
  br label %17, !dbg !1025

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1026
  br label %17, !dbg !1026

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !1027
  ret i1 %18, !dbg !1027
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20VarArg2StrIfSuitableIx9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !1028 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !1029, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !1029
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIxEENS_8EncodingEv(), !dbg !1030
  %11 = sext i8 %10 to i32, !dbg !1030
  %12 = icmp eq i32 %9, %11, !dbg !1031
  br i1 %12, label %13, label %16, !dbg !1029

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !1032, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11varying2StrIxEEDav(%class.ArgWriter* %14), !dbg !1033
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !1034
  store i1 true, i1* %4, align 1, !dbg !1035
  br label %17, !dbg !1035

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1036
  br label %17, !dbg !1036

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !1037
  ret i1 %18, !dbg !1037
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4UniformIxEENS_8EncodingEv() #25 comdat !dbg !1038 {
  ret i8 5, !dbg !1039
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11uniform2StrIxEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !1040 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i8*, align 4
  %4 = alloca <100 x i8>, align 128
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i8* @_Z16cmType2SpecifierIxEPKcv(), !dbg !1041
  store i8* %7, i8** %3, align 4, !dbg !1042, !tbaa !17
  %8 = load i8*, i8** %3, align 4, !dbg !1043, !tbaa !17
  %9 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %8, <100 x i8>* %4, i32 0, i32 99), !dbg !1044
  store i32 %9, i32* %5, align 4, !dbg !1045, !tbaa !13
  %10 = load i32, i32* %5, align 4, !dbg !1046, !tbaa !13
  %11 = trunc i32 %10 to i16, !dbg !1046
  %12 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !1047, !tbaa !7
  %13 = mul i16 %11, 1, !dbg !1047
  %14 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %12, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %13, i32 undef, i1 true), !dbg !1047
  store <100 x i8> %14, <100 x i8>* %4, align 128, !dbg !1047, !tbaa !7
  call void @_ZN9ArgWriter8WriteArgIxEEvv(%class.ArgWriter* %6), !dbg !1048
  %15 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !1049, !tbaa !7
  ret <100 x i8> %15, !dbg !1050
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_Z16cmType2SpecifierIxEPKcv() #25 comdat !dbg !1051 {
  %1 = call i8* @_ZN9PrintInfo14type2SpecifierIxEEPKcv(), !dbg !1052
  ret i8* %1, !dbg !1053
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriter8WriteArgIxEEvv(%class.ArgWriter*) #25 comdat align 2 !dbg !1054 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !1055
  store i32 %7, i32* %3, align 4, !dbg !1056, !tbaa !13
  %8 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !1057
  store i32 %8, i32* %4, align 4, !dbg !1058, !tbaa !13
  %9 = call i32 @llvm.genx.predefined.surface(i32 2), !dbg !1059
  store i32 %9, i32* %5, align 4, !dbg !1060, !tbaa !31
  %10 = load i32, i32* %5, align 4, !dbg !1061, !tbaa !31
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !1062
  %12 = load i32, i32* %11, align 8, !dbg !1062, !tbaa !59
  %13 = load i32, i32* %3, align 4, !dbg !1063, !tbaa !13
  %14 = load i32, i32* %4, align 4, !dbg !1064, !tbaa !13
  call void @_ZN7details18_cm_print_args_rawIxEEv15cm_surfaceindexjjj(i32 %10, i32 %12, i32 %13, i32 %14), !dbg !1065
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !1066
  %16 = load i32, i32* %15, align 8, !dbg !1067, !tbaa !59
  %17 = add i32 %16, 32, !dbg !1067
  store i32 %17, i32* %15, align 8, !dbg !1067, !tbaa !59
  ret void, !dbg !1068
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_ZN9PrintInfo14type2SpecifierIxEEPKcv() #25 comdat !dbg !1069 {
  ret i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.8, i32 0, i32 0), !dbg !1070
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details18_cm_print_args_rawIxEEv15cm_surfaceindexjjj(i32, i32, i32, i32) #10 comdat {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca <8 x i32>, align 32
  store i32 %0, i32* %5, align 4, !tbaa !31
  store i32 %1, i32* %6, align 4, !tbaa !13
  store i32 %2, i32* %7, align 4, !tbaa !13
  store i32 %3, i32* %8, align 4, !tbaa !13
  store <8 x i32> zeroinitializer, <8 x i32>* %9, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %11 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %10, <1 x i32> <i32 3>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %11, <8 x i32>* %9, align 32, !tbaa !7
  %12 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %13 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %12, <1 x i32> <i32 7>, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true)
  store <8 x i32> %13, <8 x i32>* %9, align 32, !tbaa !7
  %14 = load i32, i32* %8, align 4, !tbaa !13
  %15 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %16 = insertelement <1 x i32> undef, i32 %14, i32 0
  %17 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %15, <1 x i32> %16, i32 0, i32 1, i32 0, i16 28, i32 undef, i1 true)
  store <8 x i32> %17, <8 x i32>* %9, align 32, !tbaa !7
  %18 = load i32, i32* %7, align 4, !tbaa !13
  %19 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %20 = insertelement <1 x i32> undef, i32 %18, i32 0
  %21 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %19, <1 x i32> %20, i32 0, i32 1, i32 0, i16 24, i32 undef, i1 true)
  store <8 x i32> %21, <8 x i32>* %9, align 32, !tbaa !7
  %22 = load i32, i32* %5, align 4, !tbaa !31
  %23 = load i32, i32* %6, align 4, !tbaa !13
  %24 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  call void @_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_(i32 %22, i32 %23, <8 x i32> %24)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4VaryingIxEENS_8EncodingEv() #25 comdat !dbg !1071 {
  ret i8 13, !dbg !1072
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11varying2StrIxEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !1073 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1074, !tbaa !7
  %8 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %7, <1 x i8> <i8 91>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !1074
  store <100 x i8> %8, <100 x i8>* %3, align 128, !dbg !1074, !tbaa !7
  store i32 1, i32* %4, align 4, !dbg !1075, !tbaa !13
  store i32 0, i32* %5, align 4, !dbg !1076, !tbaa !13
  br label %9, !dbg !1077

; <label>:9:                                      ; preds = %58, %1
  %10 = load i32, i32* %5, align 4, !dbg !1078, !tbaa !13
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !1079
  %12 = load i32, i32* %11, align 4, !dbg !1079, !tbaa !68
  %13 = icmp slt i32 %10, %12, !dbg !1080
  br i1 %13, label %14, label %61, !dbg !1081

; <label>:14:                                     ; preds = %9
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 4, !dbg !1082
  %16 = load i64, i64* %15, align 8, !dbg !1082, !tbaa !71
  %17 = load i32, i32* %5, align 4, !dbg !1083, !tbaa !13
  %18 = zext i32 %17 to i64, !dbg !1084
  %19 = shl i64 1, %18, !dbg !1084
  %20 = and i64 %16, %19, !dbg !1085
  %21 = icmp ne i64 %20, 0, !dbg !1082
  br i1 %21, label %22, label %32, !dbg !1082

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %4, align 4, !dbg !1086, !tbaa !13
  %24 = call i32 @_Z21requiredSpace4VecElemIxL9LaneState1EEiv(), !dbg !1087
  %25 = sub i32 100, %24, !dbg !1088
  %26 = sub i32 %25, 1, !dbg !1089
  %27 = icmp uge i32 %23, %26, !dbg !1090
  br i1 %27, label %28, label %29, !dbg !1086

; <label>:28:                                     ; preds = %22
  br label %61, !dbg !1091

; <label>:29:                                     ; preds = %22
  %30 = load i32, i32* %4, align 4, !dbg !1092, !tbaa !13
  %31 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIxL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %30), !dbg !1093
  store i32 %31, i32* %4, align 4, !dbg !1094, !tbaa !13
  br label %42, !dbg !1095

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %4, align 4, !dbg !1096, !tbaa !13
  %34 = call i32 @_Z21requiredSpace4VecElemIxL9LaneState0EEiv(), !dbg !1097
  %35 = sub i32 100, %34, !dbg !1098
  %36 = sub i32 %35, 1, !dbg !1099
  %37 = icmp uge i32 %33, %36, !dbg !1100
  br i1 %37, label %38, label %39, !dbg !1096

; <label>:38:                                     ; preds = %32
  br label %61, !dbg !1101

; <label>:39:                                     ; preds = %32
  %40 = load i32, i32* %4, align 4, !dbg !1102, !tbaa !13
  %41 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIxL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %40), !dbg !1103
  store i32 %41, i32* %4, align 4, !dbg !1104, !tbaa !13
  br label %42

; <label>:42:                                     ; preds = %39, %29
  %43 = load i32, i32* %5, align 4, !dbg !1105, !tbaa !13
  %44 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !1106
  %45 = load i32, i32* %44, align 4, !dbg !1106, !tbaa !68
  %46 = sub nsw i32 %45, 1, !dbg !1107
  %47 = icmp eq i32 %43, %46, !dbg !1108
  %48 = zext i1 %47 to i64, !dbg !1105
  %49 = select i1 %47, i8 93, i8 44, !dbg !1105
  %50 = load i32, i32* %4, align 4, !dbg !1109, !tbaa !13
  %51 = trunc i32 %50 to i16, !dbg !1109
  %52 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1110, !tbaa !7
  %53 = insertelement <1 x i8> undef, i8 %49, i32 0, !dbg !1110
  %54 = mul i16 %51, 1, !dbg !1110
  %55 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %52, <1 x i8> %53, i32 0, i32 1, i32 0, i16 %54, i32 undef, i1 true), !dbg !1110
  store <100 x i8> %55, <100 x i8>* %3, align 128, !dbg !1110, !tbaa !7
  %56 = load i32, i32* %4, align 4, !dbg !1111, !tbaa !13
  %57 = add nsw i32 %56, 1, !dbg !1111
  store i32 %57, i32* %4, align 4, !dbg !1111, !tbaa !13
  call void @_ZN9ArgWriter8WriteArgIxEEvv(%class.ArgWriter* %6), !dbg !1112
  br label %58, !dbg !1113

; <label>:58:                                     ; preds = %42
  %59 = load i32, i32* %5, align 4, !dbg !1114, !tbaa !13
  %60 = add nsw i32 %59, 1, !dbg !1114
  store i32 %60, i32* %5, align 4, !dbg !1114, !tbaa !13
  br label %9, !dbg !1081, !llvm.loop !1115

; <label>:61:                                     ; preds = %38, %28, %9
  %62 = load i32, i32* %4, align 4, !dbg !1116, !tbaa !13
  %63 = trunc i32 %62 to i16, !dbg !1116
  %64 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1117, !tbaa !7
  %65 = mul i16 %63, 1, !dbg !1117
  %66 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %64, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %65, i32 undef, i1 true), !dbg !1117
  store <100 x i8> %66, <100 x i8>* %3, align 128, !dbg !1117, !tbaa !7
  %67 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1118, !tbaa !7
  ret <100 x i8> %67, !dbg !1119
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIxL9LaneState1EEiv() #25 comdat !dbg !1120 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIxEPKcv(), !dbg !1121
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !1122
  store i32 %3, i32* %1, align 4, !dbg !1123, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !1124, !tbaa !13
  %5 = add nsw i32 %4, 1, !dbg !1125
  ret i32 %5, !dbg !1126
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIxL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !1127 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIxEPKcv(), !dbg !1128
  store i8* %9, i8** %7, align 4, !dbg !1129, !tbaa !17
  %10 = load i8*, i8** %7, align 4, !dbg !1130, !tbaa !17
  %11 = load i32, i32* %6, align 4, !dbg !1131, !tbaa !13
  %12 = load i32, i32* %6, align 4, !dbg !1132, !tbaa !13
  %13 = sub i32 100, %12, !dbg !1133
  %14 = sub i32 %13, 1, !dbg !1134
  %15 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %10, <100 x i8>* %1, i32 %11, i32 %14), !dbg !1135
  %16 = load i32, i32* %6, align 4, !dbg !1136, !tbaa !13
  %17 = add nsw i32 %16, %15, !dbg !1136
  store i32 %17, i32* %6, align 4, !dbg !1136, !tbaa !13
  %18 = load i32, i32* %6, align 4, !dbg !1137, !tbaa !13
  ret i32 %18, !dbg !1138
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIxL9LaneState0EEiv() #25 comdat !dbg !1139 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIxEPKcv(), !dbg !1140
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !1141
  store i32 %3, i32* %1, align 4, !dbg !1142, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !1143, !tbaa !13
  %5 = add nsw i32 %4, 5, !dbg !1144
  ret i32 %5, !dbg !1145
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIxL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !1146 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIxEPKcv(), !dbg !1147
  store i8* %9, i8** %7, align 4, !dbg !1148, !tbaa !17
  %10 = load i32, i32* %6, align 4, !dbg !1149, !tbaa !13
  %11 = load i32, i32* %6, align 4, !dbg !1150, !tbaa !13
  %12 = sub i32 100, %11, !dbg !1151
  %13 = sub i32 %12, 1, !dbg !1152
  %14 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), <100 x i8>* %1, i32 %10, i32 %13), !dbg !1153
  %15 = load i32, i32* %6, align 4, !dbg !1154, !tbaa !13
  %16 = add nsw i32 %15, %14, !dbg !1154
  store i32 %16, i32* %6, align 4, !dbg !1154, !tbaa !13
  %17 = load i8*, i8** %7, align 4, !dbg !1155, !tbaa !17
  %18 = load i32, i32* %6, align 4, !dbg !1156, !tbaa !13
  %19 = load i32, i32* %6, align 4, !dbg !1157, !tbaa !13
  %20 = sub i32 100, %19, !dbg !1158
  %21 = sub i32 %20, 1, !dbg !1159
  %22 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %17, <100 x i8>* %1, i32 %18, i32 %21), !dbg !1160
  %23 = load i32, i32* %6, align 4, !dbg !1161, !tbaa !13
  %24 = add nsw i32 %23, %22, !dbg !1161
  store i32 %24, i32* %6, align 4, !dbg !1161, !tbaa !13
  %25 = load i32, i32* %6, align 4, !dbg !1162, !tbaa !13
  %26 = load i32, i32* %6, align 4, !dbg !1163, !tbaa !13
  %27 = sub i32 100, %26, !dbg !1164
  %28 = sub i32 %27, 1, !dbg !1165
  %29 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0), <100 x i8>* %1, i32 %25, i32 %28), !dbg !1166
  %30 = load i32, i32* %6, align 4, !dbg !1167, !tbaa !13
  %31 = add nsw i32 %30, %29, !dbg !1167
  store i32 %31, i32* %6, align 4, !dbg !1167, !tbaa !13
  %32 = load i32, i32* %6, align 4, !dbg !1168, !tbaa !13
  ret i32 %32, !dbg !1169
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20UniArg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !1170 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !1171, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !1171
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIyEENS_8EncodingEv(), !dbg !1172
  %11 = sext i8 %10 to i32, !dbg !1172
  %12 = icmp eq i32 %9, %11, !dbg !1173
  br i1 %12, label %13, label %16, !dbg !1171

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !1174, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11uniform2StrIyEEDav(%class.ArgWriter* %14), !dbg !1175
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !1176
  store i1 true, i1* %4, align 1, !dbg !1177
  br label %17, !dbg !1177

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1178
  br label %17, !dbg !1178

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !1179
  ret i1 %18, !dbg !1179
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20VarArg2StrIfSuitableIy9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !1180 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !1181, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !1181
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIyEENS_8EncodingEv(), !dbg !1182
  %11 = sext i8 %10 to i32, !dbg !1182
  %12 = icmp eq i32 %9, %11, !dbg !1183
  br i1 %12, label %13, label %16, !dbg !1181

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !1184, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11varying2StrIyEEDav(%class.ArgWriter* %14), !dbg !1185
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !1186
  store i1 true, i1* %4, align 1, !dbg !1187
  br label %17, !dbg !1187

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1188
  br label %17, !dbg !1188

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !1189
  ret i1 %18, !dbg !1189
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4UniformIyEENS_8EncodingEv() #25 comdat !dbg !1190 {
  ret i8 6, !dbg !1191
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11uniform2StrIyEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !1192 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i8*, align 4
  %4 = alloca <100 x i8>, align 128
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i8* @_Z16cmType2SpecifierIyEPKcv(), !dbg !1193
  store i8* %7, i8** %3, align 4, !dbg !1194, !tbaa !17
  %8 = load i8*, i8** %3, align 4, !dbg !1195, !tbaa !17
  %9 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %8, <100 x i8>* %4, i32 0, i32 99), !dbg !1196
  store i32 %9, i32* %5, align 4, !dbg !1197, !tbaa !13
  %10 = load i32, i32* %5, align 4, !dbg !1198, !tbaa !13
  %11 = trunc i32 %10 to i16, !dbg !1198
  %12 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !1199, !tbaa !7
  %13 = mul i16 %11, 1, !dbg !1199
  %14 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %12, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %13, i32 undef, i1 true), !dbg !1199
  store <100 x i8> %14, <100 x i8>* %4, align 128, !dbg !1199, !tbaa !7
  call void @_ZN9ArgWriter8WriteArgIyEEvv(%class.ArgWriter* %6), !dbg !1200
  %15 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !1201, !tbaa !7
  ret <100 x i8> %15, !dbg !1202
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_Z16cmType2SpecifierIyEPKcv() #25 comdat !dbg !1203 {
  %1 = call i8* @_ZN9PrintInfo14type2SpecifierIyEEPKcv(), !dbg !1204
  ret i8* %1, !dbg !1205
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriter8WriteArgIyEEvv(%class.ArgWriter*) #25 comdat align 2 !dbg !1206 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !1207
  store i32 %7, i32* %3, align 4, !dbg !1208, !tbaa !13
  %8 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !1209
  store i32 %8, i32* %4, align 4, !dbg !1210, !tbaa !13
  %9 = call i32 @llvm.genx.predefined.surface(i32 2), !dbg !1211
  store i32 %9, i32* %5, align 4, !dbg !1212, !tbaa !31
  %10 = load i32, i32* %5, align 4, !dbg !1213, !tbaa !31
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !1214
  %12 = load i32, i32* %11, align 8, !dbg !1214, !tbaa !59
  %13 = load i32, i32* %3, align 4, !dbg !1215, !tbaa !13
  %14 = load i32, i32* %4, align 4, !dbg !1216, !tbaa !13
  call void @_ZN7details18_cm_print_args_rawIyEEv15cm_surfaceindexjjj(i32 %10, i32 %12, i32 %13, i32 %14), !dbg !1217
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !1218
  %16 = load i32, i32* %15, align 8, !dbg !1219, !tbaa !59
  %17 = add i32 %16, 32, !dbg !1219
  store i32 %17, i32* %15, align 8, !dbg !1219, !tbaa !59
  ret void, !dbg !1220
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_ZN9PrintInfo14type2SpecifierIyEEPKcv() #25 comdat !dbg !1221 {
  ret i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.9, i32 0, i32 0), !dbg !1222
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details18_cm_print_args_rawIyEEv15cm_surfaceindexjjj(i32, i32, i32, i32) #10 comdat {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca <8 x i32>, align 32
  store i32 %0, i32* %5, align 4, !tbaa !31
  store i32 %1, i32* %6, align 4, !tbaa !13
  store i32 %2, i32* %7, align 4, !tbaa !13
  store i32 %3, i32* %8, align 4, !tbaa !13
  store <8 x i32> zeroinitializer, <8 x i32>* %9, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %11 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %10, <1 x i32> <i32 3>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %11, <8 x i32>* %9, align 32, !tbaa !7
  %12 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %13 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %12, <1 x i32> <i32 8>, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true)
  store <8 x i32> %13, <8 x i32>* %9, align 32, !tbaa !7
  %14 = load i32, i32* %8, align 4, !tbaa !13
  %15 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %16 = insertelement <1 x i32> undef, i32 %14, i32 0
  %17 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %15, <1 x i32> %16, i32 0, i32 1, i32 0, i16 28, i32 undef, i1 true)
  store <8 x i32> %17, <8 x i32>* %9, align 32, !tbaa !7
  %18 = load i32, i32* %7, align 4, !tbaa !13
  %19 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %20 = insertelement <1 x i32> undef, i32 %18, i32 0
  %21 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %19, <1 x i32> %20, i32 0, i32 1, i32 0, i16 24, i32 undef, i1 true)
  store <8 x i32> %21, <8 x i32>* %9, align 32, !tbaa !7
  %22 = load i32, i32* %5, align 4, !tbaa !31
  %23 = load i32, i32* %6, align 4, !tbaa !13
  %24 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  call void @_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_(i32 %22, i32 %23, <8 x i32> %24)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4VaryingIyEENS_8EncodingEv() #25 comdat !dbg !1223 {
  ret i8 14, !dbg !1224
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11varying2StrIyEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !1225 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1226, !tbaa !7
  %8 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %7, <1 x i8> <i8 91>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !1226
  store <100 x i8> %8, <100 x i8>* %3, align 128, !dbg !1226, !tbaa !7
  store i32 1, i32* %4, align 4, !dbg !1227, !tbaa !13
  store i32 0, i32* %5, align 4, !dbg !1228, !tbaa !13
  br label %9, !dbg !1229

; <label>:9:                                      ; preds = %58, %1
  %10 = load i32, i32* %5, align 4, !dbg !1230, !tbaa !13
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !1231
  %12 = load i32, i32* %11, align 4, !dbg !1231, !tbaa !68
  %13 = icmp slt i32 %10, %12, !dbg !1232
  br i1 %13, label %14, label %61, !dbg !1233

; <label>:14:                                     ; preds = %9
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 4, !dbg !1234
  %16 = load i64, i64* %15, align 8, !dbg !1234, !tbaa !71
  %17 = load i32, i32* %5, align 4, !dbg !1235, !tbaa !13
  %18 = zext i32 %17 to i64, !dbg !1236
  %19 = shl i64 1, %18, !dbg !1236
  %20 = and i64 %16, %19, !dbg !1237
  %21 = icmp ne i64 %20, 0, !dbg !1234
  br i1 %21, label %22, label %32, !dbg !1234

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %4, align 4, !dbg !1238, !tbaa !13
  %24 = call i32 @_Z21requiredSpace4VecElemIyL9LaneState1EEiv(), !dbg !1239
  %25 = sub i32 100, %24, !dbg !1240
  %26 = sub i32 %25, 1, !dbg !1241
  %27 = icmp uge i32 %23, %26, !dbg !1242
  br i1 %27, label %28, label %29, !dbg !1238

; <label>:28:                                     ; preds = %22
  br label %61, !dbg !1243

; <label>:29:                                     ; preds = %22
  %30 = load i32, i32* %4, align 4, !dbg !1244, !tbaa !13
  %31 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIyL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %30), !dbg !1245
  store i32 %31, i32* %4, align 4, !dbg !1246, !tbaa !13
  br label %42, !dbg !1247

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %4, align 4, !dbg !1248, !tbaa !13
  %34 = call i32 @_Z21requiredSpace4VecElemIyL9LaneState0EEiv(), !dbg !1249
  %35 = sub i32 100, %34, !dbg !1250
  %36 = sub i32 %35, 1, !dbg !1251
  %37 = icmp uge i32 %33, %36, !dbg !1252
  br i1 %37, label %38, label %39, !dbg !1248

; <label>:38:                                     ; preds = %32
  br label %61, !dbg !1253

; <label>:39:                                     ; preds = %32
  %40 = load i32, i32* %4, align 4, !dbg !1254, !tbaa !13
  %41 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIyL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %40), !dbg !1255
  store i32 %41, i32* %4, align 4, !dbg !1256, !tbaa !13
  br label %42

; <label>:42:                                     ; preds = %39, %29
  %43 = load i32, i32* %5, align 4, !dbg !1257, !tbaa !13
  %44 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !1258
  %45 = load i32, i32* %44, align 4, !dbg !1258, !tbaa !68
  %46 = sub nsw i32 %45, 1, !dbg !1259
  %47 = icmp eq i32 %43, %46, !dbg !1260
  %48 = zext i1 %47 to i64, !dbg !1257
  %49 = select i1 %47, i8 93, i8 44, !dbg !1257
  %50 = load i32, i32* %4, align 4, !dbg !1261, !tbaa !13
  %51 = trunc i32 %50 to i16, !dbg !1261
  %52 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1262, !tbaa !7
  %53 = insertelement <1 x i8> undef, i8 %49, i32 0, !dbg !1262
  %54 = mul i16 %51, 1, !dbg !1262
  %55 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %52, <1 x i8> %53, i32 0, i32 1, i32 0, i16 %54, i32 undef, i1 true), !dbg !1262
  store <100 x i8> %55, <100 x i8>* %3, align 128, !dbg !1262, !tbaa !7
  %56 = load i32, i32* %4, align 4, !dbg !1263, !tbaa !13
  %57 = add nsw i32 %56, 1, !dbg !1263
  store i32 %57, i32* %4, align 4, !dbg !1263, !tbaa !13
  call void @_ZN9ArgWriter8WriteArgIyEEvv(%class.ArgWriter* %6), !dbg !1264
  br label %58, !dbg !1265

; <label>:58:                                     ; preds = %42
  %59 = load i32, i32* %5, align 4, !dbg !1266, !tbaa !13
  %60 = add nsw i32 %59, 1, !dbg !1266
  store i32 %60, i32* %5, align 4, !dbg !1266, !tbaa !13
  br label %9, !dbg !1233, !llvm.loop !1267

; <label>:61:                                     ; preds = %38, %28, %9
  %62 = load i32, i32* %4, align 4, !dbg !1268, !tbaa !13
  %63 = trunc i32 %62 to i16, !dbg !1268
  %64 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1269, !tbaa !7
  %65 = mul i16 %63, 1, !dbg !1269
  %66 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %64, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %65, i32 undef, i1 true), !dbg !1269
  store <100 x i8> %66, <100 x i8>* %3, align 128, !dbg !1269, !tbaa !7
  %67 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1270, !tbaa !7
  ret <100 x i8> %67, !dbg !1271
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIyL9LaneState1EEiv() #25 comdat !dbg !1272 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIyEPKcv(), !dbg !1273
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !1274
  store i32 %3, i32* %1, align 4, !dbg !1275, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !1276, !tbaa !13
  %5 = add nsw i32 %4, 1, !dbg !1277
  ret i32 %5, !dbg !1278
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIyL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !1279 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIyEPKcv(), !dbg !1280
  store i8* %9, i8** %7, align 4, !dbg !1281, !tbaa !17
  %10 = load i8*, i8** %7, align 4, !dbg !1282, !tbaa !17
  %11 = load i32, i32* %6, align 4, !dbg !1283, !tbaa !13
  %12 = load i32, i32* %6, align 4, !dbg !1284, !tbaa !13
  %13 = sub i32 100, %12, !dbg !1285
  %14 = sub i32 %13, 1, !dbg !1286
  %15 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %10, <100 x i8>* %1, i32 %11, i32 %14), !dbg !1287
  %16 = load i32, i32* %6, align 4, !dbg !1288, !tbaa !13
  %17 = add nsw i32 %16, %15, !dbg !1288
  store i32 %17, i32* %6, align 4, !dbg !1288, !tbaa !13
  %18 = load i32, i32* %6, align 4, !dbg !1289, !tbaa !13
  ret i32 %18, !dbg !1290
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIyL9LaneState0EEiv() #25 comdat !dbg !1291 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIyEPKcv(), !dbg !1292
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !1293
  store i32 %3, i32* %1, align 4, !dbg !1294, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !1295, !tbaa !13
  %5 = add nsw i32 %4, 5, !dbg !1296
  ret i32 %5, !dbg !1297
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIyL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !1298 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIyEPKcv(), !dbg !1299
  store i8* %9, i8** %7, align 4, !dbg !1300, !tbaa !17
  %10 = load i32, i32* %6, align 4, !dbg !1301, !tbaa !13
  %11 = load i32, i32* %6, align 4, !dbg !1302, !tbaa !13
  %12 = sub i32 100, %11, !dbg !1303
  %13 = sub i32 %12, 1, !dbg !1304
  %14 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), <100 x i8>* %1, i32 %10, i32 %13), !dbg !1305
  %15 = load i32, i32* %6, align 4, !dbg !1306, !tbaa !13
  %16 = add nsw i32 %15, %14, !dbg !1306
  store i32 %16, i32* %6, align 4, !dbg !1306, !tbaa !13
  %17 = load i8*, i8** %7, align 4, !dbg !1307, !tbaa !17
  %18 = load i32, i32* %6, align 4, !dbg !1308, !tbaa !13
  %19 = load i32, i32* %6, align 4, !dbg !1309, !tbaa !13
  %20 = sub i32 100, %19, !dbg !1310
  %21 = sub i32 %20, 1, !dbg !1311
  %22 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %17, <100 x i8>* %1, i32 %18, i32 %21), !dbg !1312
  %23 = load i32, i32* %6, align 4, !dbg !1313, !tbaa !13
  %24 = add nsw i32 %23, %22, !dbg !1313
  store i32 %24, i32* %6, align 4, !dbg !1313, !tbaa !13
  %25 = load i32, i32* %6, align 4, !dbg !1314, !tbaa !13
  %26 = load i32, i32* %6, align 4, !dbg !1315, !tbaa !13
  %27 = sub i32 100, %26, !dbg !1316
  %28 = sub i32 %27, 1, !dbg !1317
  %29 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0), <100 x i8>* %1, i32 %25, i32 %28), !dbg !1318
  %30 = load i32, i32* %6, align 4, !dbg !1319, !tbaa !13
  %31 = add nsw i32 %30, %29, !dbg !1319
  store i32 %31, i32* %6, align 4, !dbg !1319, !tbaa !13
  %32 = load i32, i32* %6, align 4, !dbg !1320, !tbaa !13
  ret i32 %32, !dbg !1321
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20UniArg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !1322 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !1323, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !1323
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIdEENS_8EncodingEv(), !dbg !1324
  %11 = sext i8 %10 to i32, !dbg !1324
  %12 = icmp eq i32 %9, %11, !dbg !1325
  br i1 %12, label %13, label %16, !dbg !1323

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !1326, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11uniform2StrIdEEDav(%class.ArgWriter* %14), !dbg !1327
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !1328
  store i1 true, i1* %4, align 1, !dbg !1329
  br label %17, !dbg !1329

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1330
  br label %17, !dbg !1330

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !1331
  ret i1 %18, !dbg !1331
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20VarArg2StrIfSuitableId9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !1332 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !1333, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !1333
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIdEENS_8EncodingEv(), !dbg !1334
  %11 = sext i8 %10 to i32, !dbg !1334
  %12 = icmp eq i32 %9, %11, !dbg !1335
  br i1 %12, label %13, label %16, !dbg !1333

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !1336, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11varying2StrIdEEDav(%class.ArgWriter* %14), !dbg !1337
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !1338
  store i1 true, i1* %4, align 1, !dbg !1339
  br label %17, !dbg !1339

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1340
  br label %17, !dbg !1340

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !1341
  ret i1 %18, !dbg !1341
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4UniformIdEENS_8EncodingEv() #25 comdat !dbg !1342 {
  ret i8 7, !dbg !1343
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11uniform2StrIdEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !1344 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i8*, align 4
  %4 = alloca <100 x i8>, align 128
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i8* @_Z16cmType2SpecifierIdEPKcv(), !dbg !1345
  store i8* %7, i8** %3, align 4, !dbg !1346, !tbaa !17
  %8 = load i8*, i8** %3, align 4, !dbg !1347, !tbaa !17
  %9 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %8, <100 x i8>* %4, i32 0, i32 99), !dbg !1348
  store i32 %9, i32* %5, align 4, !dbg !1349, !tbaa !13
  %10 = load i32, i32* %5, align 4, !dbg !1350, !tbaa !13
  %11 = trunc i32 %10 to i16, !dbg !1350
  %12 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !1351, !tbaa !7
  %13 = mul i16 %11, 1, !dbg !1351
  %14 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %12, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %13, i32 undef, i1 true), !dbg !1351
  store <100 x i8> %14, <100 x i8>* %4, align 128, !dbg !1351, !tbaa !7
  call void @_ZN9ArgWriter8WriteArgIdEEvv(%class.ArgWriter* %6), !dbg !1352
  %15 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !1353, !tbaa !7
  ret <100 x i8> %15, !dbg !1354
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_Z16cmType2SpecifierIdEPKcv() #25 comdat !dbg !1355 {
  %1 = call i8* @_ZN9PrintInfo14type2SpecifierIdEEPKcv(), !dbg !1356
  ret i8* %1, !dbg !1357
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriter8WriteArgIdEEvv(%class.ArgWriter*) #25 comdat align 2 !dbg !1358 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !1359
  store i32 %7, i32* %3, align 4, !dbg !1360, !tbaa !13
  %8 = call i32 @_ZN9ArgWriter16GetElementaryArgEv(%class.ArgWriter* %6), !dbg !1361
  store i32 %8, i32* %4, align 4, !dbg !1362, !tbaa !13
  %9 = call i32 @llvm.genx.predefined.surface(i32 2), !dbg !1363
  store i32 %9, i32* %5, align 4, !dbg !1364, !tbaa !31
  %10 = load i32, i32* %5, align 4, !dbg !1365, !tbaa !31
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !1366
  %12 = load i32, i32* %11, align 8, !dbg !1366, !tbaa !59
  %13 = load i32, i32* %3, align 4, !dbg !1367, !tbaa !13
  %14 = load i32, i32* %4, align 4, !dbg !1368, !tbaa !13
  call void @_ZN7details18_cm_print_args_rawIdEEv15cm_surfaceindexjjj(i32 %10, i32 %12, i32 %13, i32 %14), !dbg !1369
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 0, !dbg !1370
  %16 = load i32, i32* %15, align 8, !dbg !1371, !tbaa !59
  %17 = add i32 %16, 32, !dbg !1371
  store i32 %17, i32* %15, align 8, !dbg !1371, !tbaa !59
  ret void, !dbg !1372
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_ZN9PrintInfo14type2SpecifierIdEEPKcv() #25 comdat !dbg !1373 {
  ret i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.7, i32 0, i32 0), !dbg !1374
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details18_cm_print_args_rawIdEEv15cm_surfaceindexjjj(i32, i32, i32, i32) #10 comdat {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca <8 x i32>, align 32
  store i32 %0, i32* %5, align 4, !tbaa !31
  store i32 %1, i32* %6, align 4, !tbaa !13
  store i32 %2, i32* %7, align 4, !tbaa !13
  store i32 %3, i32* %8, align 4, !tbaa !13
  store <8 x i32> zeroinitializer, <8 x i32>* %9, align 32, !tbaa !7
  %10 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %11 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %10, <1 x i32> <i32 3>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true)
  store <8 x i32> %11, <8 x i32>* %9, align 32, !tbaa !7
  %12 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %13 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %12, <1 x i32> <i32 9>, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true)
  store <8 x i32> %13, <8 x i32>* %9, align 32, !tbaa !7
  %14 = load i32, i32* %8, align 4, !tbaa !13
  %15 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %16 = insertelement <1 x i32> undef, i32 %14, i32 0
  %17 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %15, <1 x i32> %16, i32 0, i32 1, i32 0, i16 28, i32 undef, i1 true)
  store <8 x i32> %17, <8 x i32>* %9, align 32, !tbaa !7
  %18 = load i32, i32* %7, align 4, !tbaa !13
  %19 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  %20 = insertelement <1 x i32> undef, i32 %18, i32 0
  %21 = call <8 x i32> @llvm.genx.wrregioni.v8i32.v1i32.i16.i1(<8 x i32> %19, <1 x i32> %20, i32 0, i32 1, i32 0, i16 24, i32 undef, i1 true)
  store <8 x i32> %21, <8 x i32>* %9, align 32, !tbaa !7
  %22 = load i32, i32* %5, align 4, !tbaa !31
  %23 = load i32, i32* %6, align 4, !tbaa !13
  %24 = load <8 x i32>, <8 x i32>* %9, align 32, !tbaa !7
  call void @_Z5writeIjLi8EEv15cm_surfaceindexiu2CMvbT0__T_(i32 %22, i32 %23, <8 x i32> %24)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4VaryingIdEENS_8EncodingEv() #25 comdat !dbg !1375 {
  ret i8 15, !dbg !1376
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11varying2StrIdEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !1377 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1378, !tbaa !7
  %8 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %7, <1 x i8> <i8 91>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !1378
  store <100 x i8> %8, <100 x i8>* %3, align 128, !dbg !1378, !tbaa !7
  store i32 1, i32* %4, align 4, !dbg !1379, !tbaa !13
  store i32 0, i32* %5, align 4, !dbg !1380, !tbaa !13
  br label %9, !dbg !1381

; <label>:9:                                      ; preds = %58, %1
  %10 = load i32, i32* %5, align 4, !dbg !1382, !tbaa !13
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !1383
  %12 = load i32, i32* %11, align 4, !dbg !1383, !tbaa !68
  %13 = icmp slt i32 %10, %12, !dbg !1384
  br i1 %13, label %14, label %61, !dbg !1385

; <label>:14:                                     ; preds = %9
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 4, !dbg !1386
  %16 = load i64, i64* %15, align 8, !dbg !1386, !tbaa !71
  %17 = load i32, i32* %5, align 4, !dbg !1387, !tbaa !13
  %18 = zext i32 %17 to i64, !dbg !1388
  %19 = shl i64 1, %18, !dbg !1388
  %20 = and i64 %16, %19, !dbg !1389
  %21 = icmp ne i64 %20, 0, !dbg !1386
  br i1 %21, label %22, label %32, !dbg !1386

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %4, align 4, !dbg !1390, !tbaa !13
  %24 = call i32 @_Z21requiredSpace4VecElemIdL9LaneState1EEiv(), !dbg !1391
  %25 = sub i32 100, %24, !dbg !1392
  %26 = sub i32 %25, 1, !dbg !1393
  %27 = icmp uge i32 %23, %26, !dbg !1394
  br i1 %27, label %28, label %29, !dbg !1390

; <label>:28:                                     ; preds = %22
  br label %61, !dbg !1395

; <label>:29:                                     ; preds = %22
  %30 = load i32, i32* %4, align 4, !dbg !1396, !tbaa !13
  %31 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIdL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %30), !dbg !1397
  store i32 %31, i32* %4, align 4, !dbg !1398, !tbaa !13
  br label %42, !dbg !1399

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %4, align 4, !dbg !1400, !tbaa !13
  %34 = call i32 @_Z21requiredSpace4VecElemIdL9LaneState0EEiv(), !dbg !1401
  %35 = sub i32 100, %34, !dbg !1402
  %36 = sub i32 %35, 1, !dbg !1403
  %37 = icmp uge i32 %33, %36, !dbg !1404
  br i1 %37, label %38, label %39, !dbg !1400

; <label>:38:                                     ; preds = %32
  br label %61, !dbg !1405

; <label>:39:                                     ; preds = %32
  %40 = load i32, i32* %4, align 4, !dbg !1406, !tbaa !13
  %41 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIdL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %40), !dbg !1407
  store i32 %41, i32* %4, align 4, !dbg !1408, !tbaa !13
  br label %42

; <label>:42:                                     ; preds = %39, %29
  %43 = load i32, i32* %5, align 4, !dbg !1409, !tbaa !13
  %44 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !1410
  %45 = load i32, i32* %44, align 4, !dbg !1410, !tbaa !68
  %46 = sub nsw i32 %45, 1, !dbg !1411
  %47 = icmp eq i32 %43, %46, !dbg !1412
  %48 = zext i1 %47 to i64, !dbg !1409
  %49 = select i1 %47, i8 93, i8 44, !dbg !1409
  %50 = load i32, i32* %4, align 4, !dbg !1413, !tbaa !13
  %51 = trunc i32 %50 to i16, !dbg !1413
  %52 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1414, !tbaa !7
  %53 = insertelement <1 x i8> undef, i8 %49, i32 0, !dbg !1414
  %54 = mul i16 %51, 1, !dbg !1414
  %55 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %52, <1 x i8> %53, i32 0, i32 1, i32 0, i16 %54, i32 undef, i1 true), !dbg !1414
  store <100 x i8> %55, <100 x i8>* %3, align 128, !dbg !1414, !tbaa !7
  %56 = load i32, i32* %4, align 4, !dbg !1415, !tbaa !13
  %57 = add nsw i32 %56, 1, !dbg !1415
  store i32 %57, i32* %4, align 4, !dbg !1415, !tbaa !13
  call void @_ZN9ArgWriter8WriteArgIdEEvv(%class.ArgWriter* %6), !dbg !1416
  br label %58, !dbg !1417

; <label>:58:                                     ; preds = %42
  %59 = load i32, i32* %5, align 4, !dbg !1418, !tbaa !13
  %60 = add nsw i32 %59, 1, !dbg !1418
  store i32 %60, i32* %5, align 4, !dbg !1418, !tbaa !13
  br label %9, !dbg !1385, !llvm.loop !1419

; <label>:61:                                     ; preds = %38, %28, %9
  %62 = load i32, i32* %4, align 4, !dbg !1420, !tbaa !13
  %63 = trunc i32 %62 to i16, !dbg !1420
  %64 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1421, !tbaa !7
  %65 = mul i16 %63, 1, !dbg !1421
  %66 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %64, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %65, i32 undef, i1 true), !dbg !1421
  store <100 x i8> %66, <100 x i8>* %3, align 128, !dbg !1421, !tbaa !7
  %67 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1422, !tbaa !7
  ret <100 x i8> %67, !dbg !1423
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIdL9LaneState1EEiv() #25 comdat !dbg !1424 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIdEPKcv(), !dbg !1425
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !1426
  store i32 %3, i32* %1, align 4, !dbg !1427, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !1428, !tbaa !13
  %5 = add nsw i32 %4, 1, !dbg !1429
  ret i32 %5, !dbg !1430
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIdL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !1431 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIdEPKcv(), !dbg !1432
  store i8* %9, i8** %7, align 4, !dbg !1433, !tbaa !17
  %10 = load i8*, i8** %7, align 4, !dbg !1434, !tbaa !17
  %11 = load i32, i32* %6, align 4, !dbg !1435, !tbaa !13
  %12 = load i32, i32* %6, align 4, !dbg !1436, !tbaa !13
  %13 = sub i32 100, %12, !dbg !1437
  %14 = sub i32 %13, 1, !dbg !1438
  %15 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %10, <100 x i8>* %1, i32 %11, i32 %14), !dbg !1439
  %16 = load i32, i32* %6, align 4, !dbg !1440, !tbaa !13
  %17 = add nsw i32 %16, %15, !dbg !1440
  store i32 %17, i32* %6, align 4, !dbg !1440, !tbaa !13
  %18 = load i32, i32* %6, align 4, !dbg !1441, !tbaa !13
  ret i32 %18, !dbg !1442
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIdL9LaneState0EEiv() #25 comdat !dbg !1443 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIdEPKcv(), !dbg !1444
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !1445
  store i32 %3, i32* %1, align 4, !dbg !1446, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !1447, !tbaa !13
  %5 = add nsw i32 %4, 5, !dbg !1448
  ret i32 %5, !dbg !1449
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIdL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !1450 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIdEPKcv(), !dbg !1451
  store i8* %9, i8** %7, align 4, !dbg !1452, !tbaa !17
  %10 = load i32, i32* %6, align 4, !dbg !1453, !tbaa !13
  %11 = load i32, i32* %6, align 4, !dbg !1454, !tbaa !13
  %12 = sub i32 100, %11, !dbg !1455
  %13 = sub i32 %12, 1, !dbg !1456
  %14 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), <100 x i8>* %1, i32 %10, i32 %13), !dbg !1457
  %15 = load i32, i32* %6, align 4, !dbg !1458, !tbaa !13
  %16 = add nsw i32 %15, %14, !dbg !1458
  store i32 %16, i32* %6, align 4, !dbg !1458, !tbaa !13
  %17 = load i8*, i8** %7, align 4, !dbg !1459, !tbaa !17
  %18 = load i32, i32* %6, align 4, !dbg !1460, !tbaa !13
  %19 = load i32, i32* %6, align 4, !dbg !1461, !tbaa !13
  %20 = sub i32 100, %19, !dbg !1462
  %21 = sub i32 %20, 1, !dbg !1463
  %22 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %17, <100 x i8>* %1, i32 %18, i32 %21), !dbg !1464
  %23 = load i32, i32* %6, align 4, !dbg !1465, !tbaa !13
  %24 = add nsw i32 %23, %22, !dbg !1465
  store i32 %24, i32* %6, align 4, !dbg !1465, !tbaa !13
  %25 = load i32, i32* %6, align 4, !dbg !1466, !tbaa !13
  %26 = load i32, i32* %6, align 4, !dbg !1467, !tbaa !13
  %27 = sub i32 100, %26, !dbg !1468
  %28 = sub i32 %27, 1, !dbg !1469
  %29 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0), <100 x i8>* %1, i32 %25, i32 %28), !dbg !1470
  %30 = load i32, i32* %6, align 4, !dbg !1471, !tbaa !13
  %31 = add nsw i32 %30, %29, !dbg !1471
  store i32 %31, i32* %6, align 4, !dbg !1471, !tbaa !13
  %32 = load i32, i32* %6, align 4, !dbg !1472, !tbaa !13
  ret i32 %32, !dbg !1473
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20UniArg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !1474 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !1475, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !1475
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIPvEENS_8EncodingEv(), !dbg !1476
  %11 = sext i8 %10 to i32, !dbg !1476
  %12 = icmp eq i32 %9, %11, !dbg !1477
  br i1 %12, label %13, label %16, !dbg !1475

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !1478, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11uniform2StrIPvEEDav(%class.ArgWriter* %14), !dbg !1479
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !1480
  store i1 true, i1* %4, align 1, !dbg !1481
  br label %17, !dbg !1481

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1482
  br label %17, !dbg !1482

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !1483
  ret i1 %18, !dbg !1483
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN7details20VarArg2StrIfSuitableIPv9ArgWriterEEbcRT0_u2CMvr100_c(i8 signext, %class.ArgWriter* dereferenceable(24), <100 x i8>*) #31 comdat !dbg !1484 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  %6 = alloca %class.ArgWriter*, align 4
  %7 = alloca <100 x i8>*, align 128
  store i8 %0, i8* %5, align 1, !tbaa !7
  store %class.ArgWriter* %1, %class.ArgWriter** %6, align 4, !tbaa !17
  store <100 x i8>* %2, <100 x i8>** %7, align 128, !tbaa !7
  %8 = load i8, i8* %5, align 1, !dbg !1485, !tbaa !7
  %9 = sext i8 %8 to i32, !dbg !1485
  %10 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIPvEENS_8EncodingEv(), !dbg !1486
  %11 = sext i8 %10 to i32, !dbg !1486
  %12 = icmp eq i32 %9, %11, !dbg !1487
  br i1 %12, label %13, label %16, !dbg !1485

; <label>:13:                                     ; preds = %3
  %14 = load %class.ArgWriter*, %class.ArgWriter** %6, align 4, !dbg !1488, !tbaa !17
  %15 = call <100 x i8> @_ZN9ArgWriter11varying2StrIPvEEDav(%class.ArgWriter* %14), !dbg !1489
  call void @llvm.genx.vstore.v100i8.p0v100i8(<100 x i8> %15, <100 x i8>* %2), !dbg !1490
  store i1 true, i1* %4, align 1, !dbg !1491
  br label %17, !dbg !1491

; <label>:16:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1492
  br label %17, !dbg !1492

; <label>:17:                                     ; preds = %16, %13
  %18 = load i1, i1* %4, align 1, !dbg !1493
  ret i1 %18, !dbg !1493
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4UniformIPvEENS_8EncodingEv() #25 comdat !dbg !1494 {
  ret i8 8, !dbg !1495
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11uniform2StrIPvEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !1496 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca i8*, align 4
  %4 = alloca <100 x i8>, align 128
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = call i8* @_Z16cmType2SpecifierIPvEPKcv(), !dbg !1497
  store i8* %7, i8** %3, align 4, !dbg !1498, !tbaa !17
  %8 = load i8*, i8** %3, align 4, !dbg !1499, !tbaa !17
  %9 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %8, <100 x i8>* %4, i32 0, i32 99), !dbg !1500
  store i32 %9, i32* %5, align 4, !dbg !1501, !tbaa !13
  %10 = load i32, i32* %5, align 4, !dbg !1502, !tbaa !13
  %11 = trunc i32 %10 to i16, !dbg !1502
  %12 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !1503, !tbaa !7
  %13 = mul i16 %11, 1, !dbg !1503
  %14 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %12, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %13, i32 undef, i1 true), !dbg !1503
  store <100 x i8> %14, <100 x i8>* %4, align 128, !dbg !1503, !tbaa !7
  call void @_ZN9ArgWriter8WriteArgIPvEEvv(%class.ArgWriter* %6), !dbg !1504
  %15 = load <100 x i8>, <100 x i8>* %4, align 128, !dbg !1505, !tbaa !7
  ret <100 x i8> %15, !dbg !1506
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i8* @_Z16cmType2SpecifierIPvEPKcv() #25 comdat !dbg !1507 {
  ret i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.10, i32 0, i32 0), !dbg !1508
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN9ArgWriter8WriteArgIPvEEvv(%class.ArgWriter*) #25 comdat align 2 !dbg !1509 {
  %2 = alloca %class.ArgWriter*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %3 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  call void @_ZN9ArgWriter8WriteArgIjEEvv(%class.ArgWriter* %3), !dbg !1510
  ret void, !dbg !1511
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal signext i8 @_ZN9PrintInfo19getEncoding4VaryingIPvEENS_8EncodingEv() #25 comdat !dbg !1512 {
  ret i8 16, !dbg !1513
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <100 x i8> @_ZN9ArgWriter11varying2StrIPvEEDav(%class.ArgWriter*) #31 comdat align 2 !dbg !1514 {
  %2 = alloca %class.ArgWriter*, align 4
  %3 = alloca <100 x i8>, align 128
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %2, align 4, !tbaa !17
  %6 = load %class.ArgWriter*, %class.ArgWriter** %2, align 4
  %7 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1515, !tbaa !7
  %8 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %7, <1 x i8> <i8 91>, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !1515
  store <100 x i8> %8, <100 x i8>* %3, align 128, !dbg !1515, !tbaa !7
  store i32 1, i32* %4, align 4, !dbg !1516, !tbaa !13
  store i32 0, i32* %5, align 4, !dbg !1517, !tbaa !13
  br label %9, !dbg !1518

; <label>:9:                                      ; preds = %58, %1
  %10 = load i32, i32* %5, align 4, !dbg !1519, !tbaa !13
  %11 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !1520
  %12 = load i32, i32* %11, align 4, !dbg !1520, !tbaa !68
  %13 = icmp slt i32 %10, %12, !dbg !1521
  br i1 %13, label %14, label %61, !dbg !1522

; <label>:14:                                     ; preds = %9
  %15 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 4, !dbg !1523
  %16 = load i64, i64* %15, align 8, !dbg !1523, !tbaa !71
  %17 = load i32, i32* %5, align 4, !dbg !1524, !tbaa !13
  %18 = zext i32 %17 to i64, !dbg !1525
  %19 = shl i64 1, %18, !dbg !1525
  %20 = and i64 %16, %19, !dbg !1526
  %21 = icmp ne i64 %20, 0, !dbg !1523
  br i1 %21, label %22, label %32, !dbg !1523

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %4, align 4, !dbg !1527, !tbaa !13
  %24 = call i32 @_Z21requiredSpace4VecElemIPvL9LaneState1EEiv(), !dbg !1528
  %25 = sub i32 100, %24, !dbg !1529
  %26 = sub i32 %25, 1, !dbg !1530
  %27 = icmp uge i32 %23, %26, !dbg !1531
  br i1 %27, label %28, label %29, !dbg !1527

; <label>:28:                                     ; preds = %22
  br label %61, !dbg !1532

; <label>:29:                                     ; preds = %22
  %30 = load i32, i32* %4, align 4, !dbg !1533, !tbaa !13
  %31 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIPvL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %30), !dbg !1534
  store i32 %31, i32* %4, align 4, !dbg !1535, !tbaa !13
  br label %42, !dbg !1536

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %4, align 4, !dbg !1537, !tbaa !13
  %34 = call i32 @_Z21requiredSpace4VecElemIPvL9LaneState0EEiv(), !dbg !1538
  %35 = sub i32 100, %34, !dbg !1539
  %36 = sub i32 %35, 1, !dbg !1540
  %37 = icmp uge i32 %33, %36, !dbg !1541
  br i1 %37, label %38, label %39, !dbg !1537

; <label>:38:                                     ; preds = %32
  br label %61, !dbg !1542

; <label>:39:                                     ; preds = %32
  %40 = load i32, i32* %4, align 4, !dbg !1543, !tbaa !13
  %41 = call i32 @_ZN9ArgWriter19writeFormat4VecElemIPvL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter* %6, <100 x i8>* %3, i32 %40), !dbg !1544
  store i32 %41, i32* %4, align 4, !dbg !1545, !tbaa !13
  br label %42

; <label>:42:                                     ; preds = %39, %29
  %43 = load i32, i32* %5, align 4, !dbg !1546, !tbaa !13
  %44 = getelementptr inbounds %class.ArgWriter, %class.ArgWriter* %6, i32 0, i32 3, !dbg !1547
  %45 = load i32, i32* %44, align 4, !dbg !1547, !tbaa !68
  %46 = sub nsw i32 %45, 1, !dbg !1548
  %47 = icmp eq i32 %43, %46, !dbg !1549
  %48 = zext i1 %47 to i64, !dbg !1546
  %49 = select i1 %47, i8 93, i8 44, !dbg !1546
  %50 = load i32, i32* %4, align 4, !dbg !1550, !tbaa !13
  %51 = trunc i32 %50 to i16, !dbg !1550
  %52 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1551, !tbaa !7
  %53 = insertelement <1 x i8> undef, i8 %49, i32 0, !dbg !1551
  %54 = mul i16 %51, 1, !dbg !1551
  %55 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %52, <1 x i8> %53, i32 0, i32 1, i32 0, i16 %54, i32 undef, i1 true), !dbg !1551
  store <100 x i8> %55, <100 x i8>* %3, align 128, !dbg !1551, !tbaa !7
  %56 = load i32, i32* %4, align 4, !dbg !1552, !tbaa !13
  %57 = add nsw i32 %56, 1, !dbg !1552
  store i32 %57, i32* %4, align 4, !dbg !1552, !tbaa !13
  call void @_ZN9ArgWriter8WriteArgIPvEEvv(%class.ArgWriter* %6), !dbg !1553
  br label %58, !dbg !1554

; <label>:58:                                     ; preds = %42
  %59 = load i32, i32* %5, align 4, !dbg !1555, !tbaa !13
  %60 = add nsw i32 %59, 1, !dbg !1555
  store i32 %60, i32* %5, align 4, !dbg !1555, !tbaa !13
  br label %9, !dbg !1522, !llvm.loop !1556

; <label>:61:                                     ; preds = %38, %28, %9
  %62 = load i32, i32* %4, align 4, !dbg !1557, !tbaa !13
  %63 = trunc i32 %62 to i16, !dbg !1557
  %64 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1558, !tbaa !7
  %65 = mul i16 %63, 1, !dbg !1558
  %66 = call <100 x i8> @llvm.genx.wrregioni.v100i8.v1i8.i16.i1(<100 x i8> %64, <1 x i8> zeroinitializer, i32 0, i32 1, i32 0, i16 %65, i32 undef, i1 true), !dbg !1558
  store <100 x i8> %66, <100 x i8>* %3, align 128, !dbg !1558, !tbaa !7
  %67 = load <100 x i8>, <100 x i8>* %3, align 128, !dbg !1559, !tbaa !7
  ret <100 x i8> %67, !dbg !1560
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIPvL9LaneState1EEiv() #25 comdat !dbg !1561 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIPvEPKcv(), !dbg !1562
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !1563
  store i32 %3, i32* %1, align 4, !dbg !1564, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !1565, !tbaa !13
  %5 = add nsw i32 %4, 1, !dbg !1566
  ret i32 %5, !dbg !1567
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIPvL9LaneState1EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !1568 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIPvEPKcv(), !dbg !1569
  store i8* %9, i8** %7, align 4, !dbg !1570, !tbaa !17
  %10 = load i8*, i8** %7, align 4, !dbg !1571, !tbaa !17
  %11 = load i32, i32* %6, align 4, !dbg !1572, !tbaa !13
  %12 = load i32, i32* %6, align 4, !dbg !1573, !tbaa !13
  %13 = sub i32 100, %12, !dbg !1574
  %14 = sub i32 %13, 1, !dbg !1575
  %15 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %10, <100 x i8>* %1, i32 %11, i32 %14), !dbg !1576
  %16 = load i32, i32* %6, align 4, !dbg !1577, !tbaa !13
  %17 = add nsw i32 %16, %15, !dbg !1577
  store i32 %17, i32* %6, align 4, !dbg !1577, !tbaa !13
  %18 = load i32, i32* %6, align 4, !dbg !1578, !tbaa !13
  ret i32 %18, !dbg !1579
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_Z21requiredSpace4VecElemIPvL9LaneState0EEiv() #25 comdat !dbg !1580 {
  %1 = alloca i32, align 4
  %2 = call i8* @_Z16cmType2SpecifierIPvEPKcv(), !dbg !1581
  %3 = call i32 @_Z6strLenIPKcEiT_(i8* %2), !dbg !1582
  store i32 %3, i32* %1, align 4, !dbg !1583, !tbaa !13
  %4 = load i32, i32* %1, align 4, !dbg !1584, !tbaa !13
  %5 = add nsw i32 %4, 5, !dbg !1585
  ret i32 %5, !dbg !1586
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN9ArgWriter19writeFormat4VecElemIPvL9LaneState0EEEiu2CMvr100_ci(%class.ArgWriter*, <100 x i8>*, i32) #25 comdat align 2 !dbg !1587 {
  %4 = alloca %class.ArgWriter*, align 4
  %5 = alloca <100 x i8>*, align 128
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 4
  store %class.ArgWriter* %0, %class.ArgWriter** %4, align 4, !tbaa !17
  store <100 x i8>* %1, <100 x i8>** %5, align 128, !tbaa !7
  store i32 %2, i32* %6, align 4, !tbaa !13
  %8 = load %class.ArgWriter*, %class.ArgWriter** %4, align 4
  %9 = call i8* @_Z16cmType2SpecifierIPvEPKcv(), !dbg !1588
  store i8* %9, i8** %7, align 4, !dbg !1589, !tbaa !17
  %10 = load i32, i32* %6, align 4, !dbg !1590, !tbaa !13
  %11 = load i32, i32* %6, align 4, !dbg !1591, !tbaa !13
  %12 = sub i32 100, %11, !dbg !1592
  %13 = sub i32 %12, 1, !dbg !1593
  %14 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), <100 x i8>* %1, i32 %10, i32 %13), !dbg !1594
  %15 = load i32, i32* %6, align 4, !dbg !1595, !tbaa !13
  %16 = add nsw i32 %15, %14, !dbg !1595
  store i32 %16, i32* %6, align 4, !dbg !1595, !tbaa !13
  %17 = load i8*, i8** %7, align 4, !dbg !1596, !tbaa !17
  %18 = load i32, i32* %6, align 4, !dbg !1597, !tbaa !13
  %19 = load i32, i32* %6, align 4, !dbg !1598, !tbaa !13
  %20 = sub i32 100, %19, !dbg !1599
  %21 = sub i32 %20, 1, !dbg !1600
  %22 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* %17, <100 x i8>* %1, i32 %18, i32 %21), !dbg !1601
  %23 = load i32, i32* %6, align 4, !dbg !1602, !tbaa !13
  %24 = add nsw i32 %23, %22, !dbg !1602
  store i32 %24, i32* %6, align 4, !dbg !1602, !tbaa !13
  %25 = load i32, i32* %6, align 4, !dbg !1603, !tbaa !13
  %26 = load i32, i32* %6, align 4, !dbg !1604, !tbaa !13
  %27 = sub i32 100, %26, !dbg !1605
  %28 = sub i32 %27, 1, !dbg !1606
  %29 = call i32 @_Z12CopyFullTextILj100EEiPKcu2CMvrT__cii(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0), <100 x i8>* %1, i32 %25, i32 %28), !dbg !1607
  %30 = load i32, i32* %6, align 4, !dbg !1608, !tbaa !13
  %31 = add nsw i32 %30, %29, !dbg !1608
  store i32 %31, i32* %6, align 4, !dbg !1608, !tbaa !13
  %32 = load i32, i32* %6, align 4, !dbg !1609, !tbaa !13
  ret i32 %32, !dbg !1610
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details11CopyTillSepIJLc0EELj100ELj128EEEiu2CMvrT0__ciu2CMvrT1__cii(<100 x i8>*, i32, <128 x i8>*, i32, i32) #25 comdat !dbg !1611 {
  %6 = alloca <100 x i8>*, align 128
  %7 = alloca i32, align 4
  %8 = alloca <128 x i8>*, align 128
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store <100 x i8>* %0, <100 x i8>** %6, align 128, !tbaa !7
  store i32 %1, i32* %7, align 4, !tbaa !13
  store <128 x i8>* %2, <128 x i8>** %8, align 128, !tbaa !7
  store i32 %3, i32* %9, align 4, !tbaa !13
  store i32 %4, i32* %10, align 4, !tbaa !13
  %12 = load i32, i32* %9, align 4, !dbg !1612, !tbaa !13
  store i32 %12, i32* %11, align 4, !dbg !1613, !tbaa !13
  br label %13, !dbg !1614

; <label>:13:                                     ; preds = %27, %5
  %14 = load i32, i32* %7, align 4, !dbg !1615, !tbaa !13
  %15 = trunc i32 %14 to i16, !dbg !1615
  %16 = call <100 x i8> @llvm.genx.vload.v100i8.p0v100i8(<100 x i8>* %0), !dbg !1616
  %17 = mul i16 %15, 1, !dbg !1616
  %18 = call <1 x i8> @llvm.genx.rdregioni.v1i8.v100i8.i16(<100 x i8> %16, i32 0, i32 1, i32 0, i16 %17, i32 undef), !dbg !1616
  %19 = extractelement <1 x i8> %18, i32 0, !dbg !1616
  %20 = sext i8 %19 to i32, !dbg !1616
  %21 = icmp ne i32 %20, 0, !dbg !1617
  br i1 %21, label %22, label %25, !dbg !1618

; <label>:22:                                     ; preds = %13
  %23 = load i32, i32* %10, align 4, !dbg !1619, !tbaa !13
  %24 = icmp ne i32 %23, 0, !dbg !1619
  br label %25

; <label>:25:                                     ; preds = %22, %13
  %26 = phi i1 [ false, %13 ], [ %24, %22 ], !dbg !1620
  br i1 %26, label %27, label %44, !dbg !1614

; <label>:27:                                     ; preds = %25
  %28 = load i32, i32* %7, align 4, !dbg !1621, !tbaa !13
  %29 = add nsw i32 %28, 1, !dbg !1621
  store i32 %29, i32* %7, align 4, !dbg !1621, !tbaa !13
  %30 = trunc i32 %28 to i16, !dbg !1622
  %31 = call <100 x i8> @llvm.genx.vload.v100i8.p0v100i8(<100 x i8>* %0), !dbg !1623
  %32 = mul i16 %30, 1, !dbg !1623
  %33 = call <1 x i8> @llvm.genx.rdregioni.v1i8.v100i8.i16(<100 x i8> %31, i32 0, i32 1, i32 0, i16 %32, i32 undef), !dbg !1623
  %34 = extractelement <1 x i8> %33, i32 0, !dbg !1623
  %35 = load i32, i32* %9, align 4, !dbg !1624, !tbaa !13
  %36 = add nsw i32 %35, 1, !dbg !1624
  store i32 %36, i32* %9, align 4, !dbg !1624, !tbaa !13
  %37 = trunc i32 %35 to i16, !dbg !1625
  %38 = call <128 x i8> @llvm.genx.vload.v128i8.p0v128i8(<128 x i8>* %2), !dbg !1626
  %39 = insertelement <1 x i8> undef, i8 %34, i32 0, !dbg !1626
  %40 = mul i16 %37, 1, !dbg !1626
  %41 = call <128 x i8> @llvm.genx.wrregioni.v128i8.v1i8.i16.i1(<128 x i8> %38, <1 x i8> %39, i32 0, i32 1, i32 0, i16 %40, i32 undef, i1 true), !dbg !1626
  call void @llvm.genx.vstore.v128i8.p0v128i8(<128 x i8> %41, <128 x i8>* %2), !dbg !1626
  %42 = load i32, i32* %10, align 4, !dbg !1627, !tbaa !13
  %43 = add nsw i32 %42, -1, !dbg !1627
  store i32 %43, i32* %10, align 4, !dbg !1627, !tbaa !13
  br label %13, !dbg !1614, !llvm.loop !1628

; <label>:44:                                     ; preds = %25
  %45 = load i32, i32* %9, align 4, !dbg !1630, !tbaa !13
  %46 = load i32, i32* %11, align 4, !dbg !1631, !tbaa !13
  %47 = sub nsw i32 %45, %46, !dbg !1632
  ret i32 %47, !dbg !1633
}

; Function Attrs: nounwind readnone
declare <1 x i8> @llvm.genx.rdregioni.v1i8.v100i8.i16(<100 x i8>, i32, i32, i32, i16, i32) #2

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_Z5writeIcLi128EEv15cm_surfaceindexiu2CMvbT0__T_(i32, i32, <128 x i8>) #20 comdat {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca <128 x i8>, align 128
  %7 = alloca i32, align 4
  store i32 %0, i32* %4, align 4, !tbaa !31
  store i32 %1, i32* %5, align 4, !tbaa !13
  store <128 x i8> %2, <128 x i8>* %6, align 128, !tbaa !7
  store i32 128, i32* %7, align 4, !tbaa !13
  %8 = load i32, i32* %4, align 4, !tbaa !31
  %9 = load i32, i32* %5, align 4, !tbaa !13
  %10 = load <128 x i8>, <128 x i8>* %6, align 128, !tbaa !7
  %11 = udiv exact i32 %9, 16
  call void @llvm.genx.oword.st.v128i8(i32 %8, i32 %11, <128 x i8> %10)
  ret void
}

declare dso_local void @_ZN7details31__cm_intrinsic_impl_oword_writeIcLi128EEEv15cm_surfaceindexiu2CMvbT0__T_(i32, i32, <128 x i8>) #24

; Function Attrs: nounwind
declare void @llvm.genx.oword.st.v128i8(i32, i32, <128 x i8>) #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i32> @_ZL11vector_castILi1EjEu2CMvbT__T0_S0_(i32) #4 !dbg !1634 {
  %2 = alloca i32, align 4
  %3 = alloca <1 x i32>, align 4
  store i32 %0, i32* %2, align 4, !tbaa !13
  %4 = load i32, i32* %2, align 4, !dbg !1635, !tbaa !13
  %5 = insertelement <1 x i32> undef, i32 %4, i32 0, !dbg !1635
  %6 = shufflevector <1 x i32> %5, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1635
  store <1 x i32> %6, <1 x i32>* %3, align 4, !dbg !1636, !tbaa !7
  %7 = load <1 x i32>, <1 x i32>* %3, align 4, !dbg !1637, !tbaa !7
  ret <1 x i32> %7, !dbg !1638
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal <1 x i32> @_ZL11vector_castILi1EiEu2CMvbT__T0_S0_(i32) #4 !dbg !1639 {
  %2 = alloca i32, align 4
  %3 = alloca <1 x i32>, align 4
  store i32 %0, i32* %2, align 4, !tbaa !13
  %4 = load i32, i32* %2, align 4, !dbg !1640, !tbaa !13
  %5 = insertelement <1 x i32> undef, i32 %4, i32 0, !dbg !1640
  %6 = shufflevector <1 x i32> %5, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1640
  store <1 x i32> %6, <1 x i32>* %3, align 4, !dbg !1641, !tbaa !7
  %7 = load <1 x i32>, <1 x i32>* %3, align 4, !dbg !1642, !tbaa !7
  ret <1 x i32> %7, !dbg !1643
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details25cm_svm_scatter_write_implIiLi1EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_(<1 x i32>, <1 x i32>) #6 comdat {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x i32>, align 4
  %5 = alloca <1 x i64>, align 8
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %4, align 4, !tbaa !7
  %6 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %7 = zext <1 x i32> %6 to <1 x i64>
  store <1 x i64> %7, <1 x i64>* %5, align 8, !tbaa !7
  %8 = load <1 x i64>, <1 x i64>* %5, align 8, !tbaa !7
  %9 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  call void @llvm.genx.svm.scatter.v1i1.v1i64.v1i32(<1 x i1> <i1 true>, i32 0, <1 x i64> %8, <1 x i32> %9)
  ret void
}

declare dso_local void @_ZN7details37__cm_intrinsic_impl_svm_scatter_writeIiLi1ELi1EEEvu2CMvbT0__yu2CMvbmlT0_T1__T_(<1 x i64>, <1 x i32>) #24

; Function Attrs: nounwind
declare void @llvm.genx.svm.scatter.v1i1.v1i64.v1i32(<1 x i1>, i32, <1 x i64>, <1 x i32>) #27

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo22switchEncoding4UniformI13UniformWriterEEbNS_8EncodingET_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32) #25 comdat !dbg !1645 {
  %3 = alloca i8, align 1
  %4 = alloca %class.UniformWriter, align 32
  %5 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  store i8 %0, i8* %3, align 1, !tbaa !193
  %6 = load i8, i8* %3, align 1, !dbg !1646, !tbaa !193
  %7 = bitcast %class.UniformWriter* %4 to i8*, !dbg !1647
  %8 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1647
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %7, i8* align 32 %8, i32 64, i1 false), !dbg !1647, !tbaa.struct !197
  %9 = call zeroext i1 @_ZN9PrintInfo6detail14switchEncodingI13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET_T0_(i8 signext %6, %class.UniformWriter* byval(%class.UniformWriter) align 32 %4, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %5), !dbg !1648
  ret i1 %9, !dbg !1649
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #32

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo22switchEncoding4VaryingI13VaryingWriterEEbNS_8EncodingET_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32) #25 comdat !dbg !1650 {
  %3 = alloca i8, align 1
  %4 = alloca %class.VaryingWriter, align 32
  %5 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  store i8 %0, i8* %3, align 1, !tbaa !193
  %6 = load i8, i8* %3, align 1, !dbg !1651, !tbaa !193
  %7 = bitcast %class.VaryingWriter* %4 to i8*, !dbg !1652
  %8 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !1652
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %7, i8* align 32 %8, i32 64, i1 false), !dbg !1652, !tbaa.struct !197
  %9 = call zeroext i1 @_ZN9PrintInfo6detail14switchEncodingI13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET_T0_(i8 signext %6, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %4, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %5), !dbg !1653
  ret i1 %9, !dbg !1654
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail14switchEncodingI13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET_T0_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1655 {
  %4 = alloca i8, align 1
  %5 = alloca %class.UniformWriter, align 32
  %6 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  %7 = alloca %class.UniformWriter, align 32
  %8 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  %9 = alloca %class.UniformWriter, align 32
  %10 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  %11 = alloca %class.UniformWriter, align 32
  %12 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  %13 = alloca %class.UniformWriter, align 32
  %14 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  %15 = alloca %class.UniformWriter, align 32
  %16 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  %17 = alloca %class.UniformWriter, align 32
  %18 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  %19 = alloca %class.UniformWriter, align 32
  %20 = alloca %"struct.PrintInfo::Encoding4Uniform", align 1
  store i8 %0, i8* %4, align 1, !tbaa !193
  %21 = load i8, i8* %4, align 1, !dbg !1656, !tbaa !193
  %22 = bitcast %class.UniformWriter* %5 to i8*, !dbg !1657
  %23 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1657
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %22, i8* align 32 %23, i32 64, i1 false), !dbg !1657, !tbaa.struct !197
  %24 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIb13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext %21, %class.UniformWriter* byval(%class.UniformWriter) align 32 %5, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %6), !dbg !1658
  br i1 %24, label %60, label %25, !dbg !1659

; <label>:25:                                     ; preds = %3
  %26 = load i8, i8* %4, align 1, !dbg !1660, !tbaa !193
  %27 = bitcast %class.UniformWriter* %7 to i8*, !dbg !1661
  %28 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1661
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %27, i8* align 32 %28, i32 64, i1 false), !dbg !1661, !tbaa.struct !197
  %29 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIi13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext %26, %class.UniformWriter* byval(%class.UniformWriter) align 32 %7, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %8), !dbg !1662
  br i1 %29, label %60, label %30, !dbg !1663

; <label>:30:                                     ; preds = %25
  %31 = load i8, i8* %4, align 1, !dbg !1664, !tbaa !193
  %32 = bitcast %class.UniformWriter* %9 to i8*, !dbg !1665
  %33 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1665
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %32, i8* align 32 %33, i32 64, i1 false), !dbg !1665, !tbaa.struct !197
  %34 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIj13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext %31, %class.UniformWriter* byval(%class.UniformWriter) align 32 %9, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %10), !dbg !1666
  br i1 %34, label %60, label %35, !dbg !1667

; <label>:35:                                     ; preds = %30
  %36 = load i8, i8* %4, align 1, !dbg !1668, !tbaa !193
  %37 = bitcast %class.UniformWriter* %11 to i8*, !dbg !1669
  %38 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1669
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %37, i8* align 32 %38, i32 64, i1 false), !dbg !1669, !tbaa.struct !197
  %39 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIf13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext %36, %class.UniformWriter* byval(%class.UniformWriter) align 32 %11, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %12), !dbg !1670
  br i1 %39, label %60, label %40, !dbg !1671

; <label>:40:                                     ; preds = %35
  %41 = load i8, i8* %4, align 1, !dbg !1672, !tbaa !193
  %42 = bitcast %class.UniformWriter* %13 to i8*, !dbg !1673
  %43 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1673
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %42, i8* align 32 %43, i32 64, i1 false), !dbg !1673, !tbaa.struct !197
  %44 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIx13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext %41, %class.UniformWriter* byval(%class.UniformWriter) align 32 %13, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %14), !dbg !1674
  br i1 %44, label %60, label %45, !dbg !1675

; <label>:45:                                     ; preds = %40
  %46 = load i8, i8* %4, align 1, !dbg !1676, !tbaa !193
  %47 = bitcast %class.UniformWriter* %15 to i8*, !dbg !1677
  %48 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1677
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %47, i8* align 32 %48, i32 64, i1 false), !dbg !1677, !tbaa.struct !197
  %49 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIy13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext %46, %class.UniformWriter* byval(%class.UniformWriter) align 32 %15, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %16), !dbg !1678
  br i1 %49, label %60, label %50, !dbg !1679

; <label>:50:                                     ; preds = %45
  %51 = load i8, i8* %4, align 1, !dbg !1680, !tbaa !193
  %52 = bitcast %class.UniformWriter* %17 to i8*, !dbg !1681
  %53 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1681
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %52, i8* align 32 %53, i32 64, i1 false), !dbg !1681, !tbaa.struct !197
  %54 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingId13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext %51, %class.UniformWriter* byval(%class.UniformWriter) align 32 %17, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %18), !dbg !1682
  br i1 %54, label %60, label %55, !dbg !1683

; <label>:55:                                     ; preds = %50
  %56 = load i8, i8* %4, align 1, !dbg !1684, !tbaa !193
  %57 = bitcast %class.UniformWriter* %19 to i8*, !dbg !1685
  %58 = bitcast %class.UniformWriter* %1 to i8*, !dbg !1685
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %57, i8* align 32 %58, i32 64, i1 false), !dbg !1685, !tbaa.struct !197
  %59 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIPv13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext %56, %class.UniformWriter* byval(%class.UniformWriter) align 32 %19, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1 %20), !dbg !1686
  br label %60, !dbg !1683

; <label>:60:                                     ; preds = %55, %50, %45, %40, %35, %30, %25, %3
  %61 = phi i1 [ true, %50 ], [ true, %45 ], [ true, %40 ], [ true, %35 ], [ true, %30 ], [ true, %25 ], [ true, %3 ], [ %59, %55 ]
  ret i1 %61, !dbg !1687
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIb13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1688 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !1689, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !1689
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIbEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"* %2), !dbg !1690
  %9 = sext i8 %8 to i32, !dbg !1691
  %10 = icmp eq i32 %7, %9, !dbg !1692
  br i1 %10, label %11, label %12, !dbg !1689

; <label>:11:                                     ; preds = %3
  call void @_ZN13UniformWriter4callIbEEvv(%class.UniformWriter* %1), !dbg !1693
  store i1 true, i1* %4, align 1, !dbg !1694
  br label %13, !dbg !1694

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1695
  br label %13, !dbg !1695

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !1696
  ret i1 %14, !dbg !1696
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIi13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1697 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !1698, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !1698
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIiEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"* %2), !dbg !1699
  %9 = sext i8 %8 to i32, !dbg !1700
  %10 = icmp eq i32 %7, %9, !dbg !1701
  br i1 %10, label %11, label %12, !dbg !1698

; <label>:11:                                     ; preds = %3
  call void @_ZN13UniformWriter4callIiEEvv(%class.UniformWriter* %1), !dbg !1702
  store i1 true, i1* %4, align 1, !dbg !1703
  br label %13, !dbg !1703

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1704
  br label %13, !dbg !1704

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !1705
  ret i1 %14, !dbg !1705
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIj13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1706 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !1707, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !1707
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIjEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"* %2), !dbg !1708
  %9 = sext i8 %8 to i32, !dbg !1709
  %10 = icmp eq i32 %7, %9, !dbg !1710
  br i1 %10, label %11, label %12, !dbg !1707

; <label>:11:                                     ; preds = %3
  call void @_ZN13UniformWriter4callIjEEvv(%class.UniformWriter* %1), !dbg !1711
  store i1 true, i1* %4, align 1, !dbg !1712
  br label %13, !dbg !1712

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1713
  br label %13, !dbg !1713

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !1714
  ret i1 %14, !dbg !1714
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIf13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1715 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !1716, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !1716
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIfEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"* %2), !dbg !1717
  %9 = sext i8 %8 to i32, !dbg !1718
  %10 = icmp eq i32 %7, %9, !dbg !1719
  br i1 %10, label %11, label %12, !dbg !1716

; <label>:11:                                     ; preds = %3
  call void @_ZN13UniformWriter4callIfEEvv(%class.UniformWriter* %1), !dbg !1720
  store i1 true, i1* %4, align 1, !dbg !1721
  br label %13, !dbg !1721

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1722
  br label %13, !dbg !1722

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !1723
  ret i1 %14, !dbg !1723
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIx13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1724 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !1725, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !1725
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIxEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"* %2), !dbg !1726
  %9 = sext i8 %8 to i32, !dbg !1727
  %10 = icmp eq i32 %7, %9, !dbg !1728
  br i1 %10, label %11, label %12, !dbg !1725

; <label>:11:                                     ; preds = %3
  call void @_ZN13UniformWriter4callIxEEvv(%class.UniformWriter* %1), !dbg !1729
  store i1 true, i1* %4, align 1, !dbg !1730
  br label %13, !dbg !1730

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1731
  br label %13, !dbg !1731

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !1732
  ret i1 %14, !dbg !1732
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIy13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1733 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !1734, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !1734
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIyEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"* %2), !dbg !1735
  %9 = sext i8 %8 to i32, !dbg !1736
  %10 = icmp eq i32 %7, %9, !dbg !1737
  br i1 %10, label %11, label %12, !dbg !1734

; <label>:11:                                     ; preds = %3
  call void @_ZN13UniformWriter4callIyEEvv(%class.UniformWriter* %1), !dbg !1738
  store i1 true, i1* %4, align 1, !dbg !1739
  br label %13, !dbg !1739

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1740
  br label %13, !dbg !1740

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !1741
  ret i1 %14, !dbg !1741
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingId13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1742 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !1743, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !1743
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIdEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"* %2), !dbg !1744
  %9 = sext i8 %8 to i32, !dbg !1745
  %10 = icmp eq i32 %7, %9, !dbg !1746
  br i1 %10, label %11, label %12, !dbg !1743

; <label>:11:                                     ; preds = %3
  call void @_ZN13UniformWriter4callIdEEvv(%class.UniformWriter* %1), !dbg !1747
  store i1 true, i1* %4, align 1, !dbg !1748
  br label %13, !dbg !1748

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1749
  br label %13, !dbg !1749

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !1750
  ret i1 %14, !dbg !1750
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIPv13UniformWriterNS_16Encoding4UniformEEEbNS_8EncodingET0_T1_(i8 signext, %class.UniformWriter* byval(%class.UniformWriter) align 32, %"struct.PrintInfo::Encoding4Uniform"* byval(%"struct.PrintInfo::Encoding4Uniform") align 1) #25 comdat !dbg !1751 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !1752, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !1752
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIPvEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"* %2), !dbg !1753
  %9 = sext i8 %8 to i32, !dbg !1754
  %10 = icmp eq i32 %7, %9, !dbg !1755
  br i1 %10, label %11, label %12, !dbg !1752

; <label>:11:                                     ; preds = %3
  call void @_ZN13UniformWriter4callIPvEEvv(%class.UniformWriter* %1), !dbg !1756
  store i1 true, i1* %4, align 1, !dbg !1757
  br label %13, !dbg !1757

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !1758
  br label %13, !dbg !1758

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !1759
  ret i1 %14, !dbg !1759
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIbEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"*) #33 comdat align 2 !dbg !1760 {
  %2 = alloca %"struct.PrintInfo::Encoding4Uniform"*, align 4
  store %"struct.PrintInfo::Encoding4Uniform"* %0, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Uniform"*, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIbEENS_8EncodingEv(), !dbg !1761
  ret i8 %4, !dbg !1762
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter4callIbEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1763 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  call void @_ZN13UniformWriter8WriteArgIbEEvv(%class.UniformWriter* %3), !dbg !1764
  ret void, !dbg !1765
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter8WriteArgIbEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1766 {
  %2 = alloca %class.UniformWriter*, align 4
  %3 = alloca i32, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %4 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %5 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %4), !dbg !1767
  %6 = icmp ne i32 %5, 0, !dbg !1767
  br i1 %6, label %7, label %12, !dbg !1767

; <label>:7:                                      ; preds = %1
  %8 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %4, i32 0, i32 4, !dbg !1768
  %9 = load <5 x i32>, <5 x i32>* %8, align 32, !dbg !1768, !tbaa !218
  %10 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %9, i32 0, i32 1, i32 0, i16 16, i32 undef), !dbg !1768
  %11 = extractelement <1 x i32> %10, i32 0, !dbg !1768
  store i32 %11, i32* %3, align 4, !dbg !1769, !tbaa !13
  br label %17, !dbg !1770

; <label>:12:                                     ; preds = %1
  %13 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %4, i32 0, i32 4, !dbg !1771
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !1771, !tbaa !218
  %15 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %14, i32 0, i32 1, i32 0, i16 12, i32 undef), !dbg !1771
  %16 = extractelement <1 x i32> %15, i32 0, !dbg !1771
  store i32 %16, i32* %3, align 4, !dbg !1772, !tbaa !13
  br label %17

; <label>:17:                                     ; preds = %12, %7
  %18 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %4, i32 0, i32 0, !dbg !1773
  %19 = load i32*, i32** %18, align 32, !dbg !1773, !tbaa !1774
  %20 = load i32, i32* %3, align 4, !dbg !1775, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %19, i32 %20, i32 0), !dbg !1776
  ret void, !dbg !1777
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter*) #25 comdat align 2 !dbg !1778 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %4 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %3, i32 0, i32 1, !dbg !1779
  %5 = load i32**, i32*** %4, align 4, !dbg !1779, !tbaa !1780
  %6 = load i32*, i32** %5, align 4, !dbg !1781, !tbaa !17
  %7 = getelementptr inbounds i32, i32* %6, i32 1, !dbg !1781
  store i32* %7, i32** %5, align 4, !dbg !1781, !tbaa !17
  %8 = load i32, i32* %6, align 4, !dbg !1782, !tbaa !13
  ret i32 %8, !dbg !1783
}

; Function Attrs: nounwind readnone
declare <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32>, i32, i32, i32, i16, i32) #2

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4), i32, i32) #4 !dbg !1784 {
  %4 = alloca i32*, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <1 x i32>, align 4
  store i32* %0, i32** %4, align 4, !tbaa !17
  store i32 %1, i32* %5, align 4, !tbaa !13
  store i32 %2, i32* %6, align 4, !tbaa !13
  %9 = call i32 @_ZN7details18_cm_print_type_oclIRA1_KcEENS_18SHADER_PRINTF_TYPEEv(), !dbg !1785
  %10 = insertelement <1 x i32> undef, i32 %9, i32 0, !dbg !1785
  %11 = shufflevector <1 x i32> %10, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1785
  store <1 x i32> %11, <1 x i32>* %7, align 4, !dbg !1786, !tbaa !7
  %12 = load i32*, i32** %4, align 4, !dbg !1787, !tbaa !17
  %13 = load i32, i32* %12, align 4, !dbg !1787, !tbaa !13
  %14 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %13), !dbg !1788
  %15 = load <1 x i32>, <1 x i32>* %7, align 4, !dbg !1789, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %14, <1 x i32> %15), !dbg !1790
  %16 = load i32*, i32** %4, align 4, !dbg !1791, !tbaa !17
  %17 = load i32, i32* %16, align 4, !dbg !1792, !tbaa !13
  %18 = add i32 %17, 4, !dbg !1792
  store i32 %18, i32* %16, align 4, !dbg !1792, !tbaa !13
  %19 = load i32, i32* %5, align 4, !dbg !1793, !tbaa !13
  %20 = insertelement <1 x i32> undef, i32 %19, i32 0, !dbg !1793
  %21 = shufflevector <1 x i32> %20, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1793
  store <1 x i32> %21, <1 x i32>* %8, align 4, !dbg !1794, !tbaa !7
  %22 = load i32*, i32** %4, align 4, !dbg !1795, !tbaa !17
  %23 = load i32, i32* %22, align 4, !dbg !1795, !tbaa !13
  %24 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %23), !dbg !1796
  %25 = load <1 x i32>, <1 x i32>* %8, align 4, !dbg !1797, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %24, <1 x i32> %25), !dbg !1798
  %26 = load i32*, i32** %4, align 4, !dbg !1799, !tbaa !17
  %27 = load i32, i32* %26, align 4, !dbg !1800, !tbaa !13
  %28 = add i32 %27, 4, !dbg !1800
  store i32 %28, i32* %26, align 4, !dbg !1800, !tbaa !13
  ret void, !dbg !1801
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details18_cm_print_type_oclIRA1_KcEENS_18SHADER_PRINTF_TYPEEv() #25 comdat {
  ret i32 5
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32>, <1 x i32>) #4 comdat {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x i32>, align 4
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %4, align 4, !tbaa !7
  %5 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %6 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  call void @_ZN7details25cm_svm_scatter_write_implIjLi1EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %5, <1 x i32> %6)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details25cm_svm_scatter_write_implIjLi1EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_(<1 x i32>, <1 x i32>) #6 comdat {
  %3 = alloca <1 x i32>, align 4
  %4 = alloca <1 x i32>, align 4
  %5 = alloca <1 x i64>, align 8
  store <1 x i32> %0, <1 x i32>* %3, align 4, !tbaa !7
  store <1 x i32> %1, <1 x i32>* %4, align 4, !tbaa !7
  %6 = load <1 x i32>, <1 x i32>* %3, align 4, !tbaa !7
  %7 = zext <1 x i32> %6 to <1 x i64>
  store <1 x i64> %7, <1 x i64>* %5, align 8, !tbaa !7
  %8 = load <1 x i64>, <1 x i64>* %5, align 8, !tbaa !7
  %9 = load <1 x i32>, <1 x i32>* %4, align 4, !tbaa !7
  call void @llvm.genx.svm.scatter.v1i1.v1i64.v1i32(<1 x i1> <i1 true>, i32 0, <1 x i64> %8, <1 x i32> %9)
  ret void
}

declare dso_local void @_ZN7details37__cm_intrinsic_impl_svm_scatter_writeIjLi1ELi1EEEvu2CMvbT0__yu2CMvbmlT0_T1__T_(<1 x i64>, <1 x i32>) #24

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIiEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"*) #33 comdat align 2 !dbg !1802 {
  %2 = alloca %"struct.PrintInfo::Encoding4Uniform"*, align 4
  store %"struct.PrintInfo::Encoding4Uniform"* %0, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Uniform"*, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIiEENS_8EncodingEv(), !dbg !1803
  ret i8 %4, !dbg !1804
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter4callIiEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1805 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  call void @_ZN13UniformWriter8WriteArgIiEEvv(%class.UniformWriter* %3), !dbg !1806
  ret void, !dbg !1807
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter8WriteArgIiEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1808 {
  %2 = alloca %class.UniformWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %5 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %6 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1809
  store i32 %6, i32* %3, align 4, !dbg !1810, !tbaa !13
  store i32 0, i32* %4, align 4, !dbg !1811, !tbaa !13
  %7 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %5, i32 0, i32 0, !dbg !1812
  %8 = load i32*, i32** %7, align 32, !dbg !1812, !tbaa !1774
  %9 = load i32, i32* %3, align 4, !dbg !1813, !tbaa !13
  %10 = load i32, i32* %4, align 4, !dbg !1814, !tbaa !13
  call void @_ZL24write_arg_with_promotionIiEvRjjj(i32* dereferenceable(4) %8, i32 %9, i32 %10), !dbg !1815
  ret void, !dbg !1816
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZL24write_arg_with_promotionIiEvRjjj(i32* dereferenceable(4), i32, i32) #4 !dbg !1817 {
  %4 = alloca i32*, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <1 x i32>, align 4
  store i32* %0, i32** %4, align 4, !tbaa !17
  store i32 %1, i32* %5, align 4, !tbaa !13
  store i32 %2, i32* %6, align 4, !tbaa !13
  %9 = call i32 @_ZN7details18_cm_print_type_oclIiEENS_18SHADER_PRINTF_TYPEEv(), !dbg !1818
  %10 = insertelement <1 x i32> undef, i32 %9, i32 0, !dbg !1818
  %11 = shufflevector <1 x i32> %10, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1818
  store <1 x i32> %11, <1 x i32>* %7, align 4, !dbg !1819, !tbaa !7
  %12 = load i32*, i32** %4, align 4, !dbg !1820, !tbaa !17
  %13 = load i32, i32* %12, align 4, !dbg !1820, !tbaa !13
  %14 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %13), !dbg !1821
  %15 = load <1 x i32>, <1 x i32>* %7, align 4, !dbg !1822, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %14, <1 x i32> %15), !dbg !1823
  %16 = load i32*, i32** %4, align 4, !dbg !1824, !tbaa !17
  %17 = load i32, i32* %16, align 4, !dbg !1825, !tbaa !13
  %18 = add i32 %17, 4, !dbg !1825
  store i32 %18, i32* %16, align 4, !dbg !1825, !tbaa !13
  %19 = load i32, i32* %5, align 4, !dbg !1826, !tbaa !13
  %20 = insertelement <1 x i32> undef, i32 %19, i32 0, !dbg !1826
  %21 = shufflevector <1 x i32> %20, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1826
  store <1 x i32> %21, <1 x i32>* %8, align 4, !dbg !1827, !tbaa !7
  %22 = load i32*, i32** %4, align 4, !dbg !1828, !tbaa !17
  %23 = load i32, i32* %22, align 4, !dbg !1828, !tbaa !13
  %24 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %23), !dbg !1829
  %25 = load <1 x i32>, <1 x i32>* %8, align 4, !dbg !1830, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %24, <1 x i32> %25), !dbg !1831
  %26 = load i32*, i32** %4, align 4, !dbg !1832, !tbaa !17
  %27 = load i32, i32* %26, align 4, !dbg !1833, !tbaa !13
  %28 = add i32 %27, 4, !dbg !1833
  store i32 %28, i32* %26, align 4, !dbg !1833, !tbaa !13
  ret void, !dbg !1834
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details18_cm_print_type_oclIiEENS_18SHADER_PRINTF_TYPEEv() #25 comdat {
  ret i32 3
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIjEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"*) #33 comdat align 2 !dbg !1835 {
  %2 = alloca %"struct.PrintInfo::Encoding4Uniform"*, align 4
  store %"struct.PrintInfo::Encoding4Uniform"* %0, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Uniform"*, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIjEENS_8EncodingEv(), !dbg !1836
  ret i8 %4, !dbg !1837
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter4callIjEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1838 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  call void @_ZN13UniformWriter8WriteArgIjEEvv(%class.UniformWriter* %3), !dbg !1839
  ret void, !dbg !1840
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter8WriteArgIjEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1841 {
  %2 = alloca %class.UniformWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %5 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %6 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1842
  store i32 %6, i32* %3, align 4, !dbg !1843, !tbaa !13
  store i32 0, i32* %4, align 4, !dbg !1844, !tbaa !13
  %7 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %5, i32 0, i32 0, !dbg !1845
  %8 = load i32*, i32** %7, align 32, !dbg !1845, !tbaa !1774
  %9 = load i32, i32* %3, align 4, !dbg !1846, !tbaa !13
  %10 = load i32, i32* %4, align 4, !dbg !1847, !tbaa !13
  call void @_ZL24write_arg_with_promotionIjEvRjjj(i32* dereferenceable(4) %8, i32 %9, i32 %10), !dbg !1848
  ret void, !dbg !1849
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZL24write_arg_with_promotionIjEvRjjj(i32* dereferenceable(4), i32, i32) #4 !dbg !1850 {
  %4 = alloca i32*, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <1 x i32>, align 4
  store i32* %0, i32** %4, align 4, !tbaa !17
  store i32 %1, i32* %5, align 4, !tbaa !13
  store i32 %2, i32* %6, align 4, !tbaa !13
  %9 = call i32 @_ZN7details18_cm_print_type_oclIjEENS_18SHADER_PRINTF_TYPEEv(), !dbg !1851
  %10 = insertelement <1 x i32> undef, i32 %9, i32 0, !dbg !1851
  %11 = shufflevector <1 x i32> %10, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1851
  store <1 x i32> %11, <1 x i32>* %7, align 4, !dbg !1852, !tbaa !7
  %12 = load i32*, i32** %4, align 4, !dbg !1853, !tbaa !17
  %13 = load i32, i32* %12, align 4, !dbg !1853, !tbaa !13
  %14 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %13), !dbg !1854
  %15 = load <1 x i32>, <1 x i32>* %7, align 4, !dbg !1855, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %14, <1 x i32> %15), !dbg !1856
  %16 = load i32*, i32** %4, align 4, !dbg !1857, !tbaa !17
  %17 = load i32, i32* %16, align 4, !dbg !1858, !tbaa !13
  %18 = add i32 %17, 4, !dbg !1858
  store i32 %18, i32* %16, align 4, !dbg !1858, !tbaa !13
  %19 = load i32, i32* %5, align 4, !dbg !1859, !tbaa !13
  %20 = insertelement <1 x i32> undef, i32 %19, i32 0, !dbg !1859
  %21 = shufflevector <1 x i32> %20, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1859
  store <1 x i32> %21, <1 x i32>* %8, align 4, !dbg !1860, !tbaa !7
  %22 = load i32*, i32** %4, align 4, !dbg !1861, !tbaa !17
  %23 = load i32, i32* %22, align 4, !dbg !1861, !tbaa !13
  %24 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %23), !dbg !1862
  %25 = load <1 x i32>, <1 x i32>* %8, align 4, !dbg !1863, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %24, <1 x i32> %25), !dbg !1864
  %26 = load i32*, i32** %4, align 4, !dbg !1865, !tbaa !17
  %27 = load i32, i32* %26, align 4, !dbg !1866, !tbaa !13
  %28 = add i32 %27, 4, !dbg !1866
  store i32 %28, i32* %26, align 4, !dbg !1866, !tbaa !13
  ret void, !dbg !1867
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details18_cm_print_type_oclIjEENS_18SHADER_PRINTF_TYPEEv() #25 comdat {
  ret i32 3
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIfEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"*) #33 comdat align 2 !dbg !1868 {
  %2 = alloca %"struct.PrintInfo::Encoding4Uniform"*, align 4
  store %"struct.PrintInfo::Encoding4Uniform"* %0, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Uniform"*, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIfEENS_8EncodingEv(), !dbg !1869
  ret i8 %4, !dbg !1870
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter4callIfEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1871 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  call void @_ZN13UniformWriter8WriteArgIfEEvv(%class.UniformWriter* %3), !dbg !1872
  ret void, !dbg !1873
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter8WriteArgIfEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1874 {
  %2 = alloca %class.UniformWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %5 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %6 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1875
  store i32 %6, i32* %3, align 4, !dbg !1876, !tbaa !13
  store i32 0, i32* %4, align 4, !dbg !1877, !tbaa !13
  %7 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %5, i32 0, i32 0, !dbg !1878
  %8 = load i32*, i32** %7, align 32, !dbg !1878, !tbaa !1774
  %9 = load i32, i32* %3, align 4, !dbg !1879, !tbaa !13
  %10 = load i32, i32* %4, align 4, !dbg !1880, !tbaa !13
  call void @_ZL24write_arg_with_promotionIfEvRjjj(i32* dereferenceable(4) %8, i32 %9, i32 %10), !dbg !1881
  ret void, !dbg !1882
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZL24write_arg_with_promotionIfEvRjjj(i32* dereferenceable(4), i32, i32) #4 !dbg !1883 {
  %4 = alloca i32*, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <1 x i32>, align 4
  store i32* %0, i32** %4, align 4, !tbaa !17
  store i32 %1, i32* %5, align 4, !tbaa !13
  store i32 %2, i32* %6, align 4, !tbaa !13
  %9 = call i32 @_ZN7details18_cm_print_type_oclIfEENS_18SHADER_PRINTF_TYPEEv(), !dbg !1884
  %10 = insertelement <1 x i32> undef, i32 %9, i32 0, !dbg !1884
  %11 = shufflevector <1 x i32> %10, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1884
  store <1 x i32> %11, <1 x i32>* %7, align 4, !dbg !1885, !tbaa !7
  %12 = load i32*, i32** %4, align 4, !dbg !1886, !tbaa !17
  %13 = load i32, i32* %12, align 4, !dbg !1886, !tbaa !13
  %14 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %13), !dbg !1887
  %15 = load <1 x i32>, <1 x i32>* %7, align 4, !dbg !1888, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %14, <1 x i32> %15), !dbg !1889
  %16 = load i32*, i32** %4, align 4, !dbg !1890, !tbaa !17
  %17 = load i32, i32* %16, align 4, !dbg !1891, !tbaa !13
  %18 = add i32 %17, 4, !dbg !1891
  store i32 %18, i32* %16, align 4, !dbg !1891, !tbaa !13
  %19 = load i32, i32* %5, align 4, !dbg !1892, !tbaa !13
  %20 = insertelement <1 x i32> undef, i32 %19, i32 0, !dbg !1892
  %21 = shufflevector <1 x i32> %20, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1892
  store <1 x i32> %21, <1 x i32>* %8, align 4, !dbg !1893, !tbaa !7
  %22 = load i32*, i32** %4, align 4, !dbg !1894, !tbaa !17
  %23 = load i32, i32* %22, align 4, !dbg !1894, !tbaa !13
  %24 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %23), !dbg !1895
  %25 = load <1 x i32>, <1 x i32>* %8, align 4, !dbg !1896, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %24, <1 x i32> %25), !dbg !1897
  %26 = load i32*, i32** %4, align 4, !dbg !1898, !tbaa !17
  %27 = load i32, i32* %26, align 4, !dbg !1899, !tbaa !13
  %28 = add i32 %27, 4, !dbg !1899
  store i32 %28, i32* %26, align 4, !dbg !1899, !tbaa !13
  ret void, !dbg !1900
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details18_cm_print_type_oclIfEENS_18SHADER_PRINTF_TYPEEv() #25 comdat {
  ret i32 4
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIxEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"*) #33 comdat align 2 !dbg !1901 {
  %2 = alloca %"struct.PrintInfo::Encoding4Uniform"*, align 4
  store %"struct.PrintInfo::Encoding4Uniform"* %0, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Uniform"*, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIxEENS_8EncodingEv(), !dbg !1902
  ret i8 %4, !dbg !1903
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter4callIxEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1904 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  call void @_ZN13UniformWriter8WriteArgIxEEvv(%class.UniformWriter* %3), !dbg !1905
  ret void, !dbg !1906
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter8WriteArgIxEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1907 {
  %2 = alloca %class.UniformWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %5 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %6 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1908
  store i32 %6, i32* %3, align 4, !dbg !1909, !tbaa !13
  %7 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1910
  store i32 %7, i32* %4, align 4, !dbg !1911, !tbaa !13
  %8 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %5, i32 0, i32 0, !dbg !1912
  %9 = load i32*, i32** %8, align 32, !dbg !1912, !tbaa !1774
  %10 = load i32, i32* %3, align 4, !dbg !1913, !tbaa !13
  %11 = load i32, i32* %4, align 4, !dbg !1914, !tbaa !13
  call void @_ZL24write_arg_with_promotionIxEvRjjj(i32* dereferenceable(4) %9, i32 %10, i32 %11), !dbg !1915
  ret void, !dbg !1916
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZL24write_arg_with_promotionIxEvRjjj(i32* dereferenceable(4), i32, i32) #6 !dbg !1917 {
  %4 = alloca i32*, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <2 x i32>, align 8
  %9 = alloca <2 x i32>, align 8
  store i32* %0, i32** %4, align 4, !tbaa !17
  store i32 %1, i32* %5, align 4, !tbaa !13
  store i32 %2, i32* %6, align 4, !tbaa !13
  %10 = call i32 @_ZN7details18_cm_print_type_oclIxEENS_18SHADER_PRINTF_TYPEEv(), !dbg !1918
  %11 = insertelement <1 x i32> undef, i32 %10, i32 0, !dbg !1918
  %12 = shufflevector <1 x i32> %11, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1918
  store <1 x i32> %12, <1 x i32>* %7, align 4, !dbg !1919, !tbaa !7
  %13 = load i32*, i32** %4, align 4, !dbg !1920, !tbaa !17
  %14 = load i32, i32* %13, align 4, !dbg !1920, !tbaa !13
  %15 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %14), !dbg !1921
  %16 = load <1 x i32>, <1 x i32>* %7, align 4, !dbg !1922, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %15, <1 x i32> %16), !dbg !1923
  %17 = load i32*, i32** %4, align 4, !dbg !1924, !tbaa !17
  %18 = load i32, i32* %17, align 4, !dbg !1925, !tbaa !13
  %19 = add i32 %18, 4, !dbg !1925
  store i32 %19, i32* %17, align 4, !dbg !1925, !tbaa !13
  %20 = load i32, i32* %5, align 4, !dbg !1926, !tbaa !13
  %21 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !1927, !tbaa !7
  %22 = insertelement <1 x i32> undef, i32 %20, i32 0, !dbg !1927
  %23 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %21, <1 x i32> %22, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !1927
  store <2 x i32> %23, <2 x i32>* %8, align 8, !dbg !1927, !tbaa !7
  %24 = load i32, i32* %6, align 4, !dbg !1928, !tbaa !13
  %25 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !1929, !tbaa !7
  %26 = insertelement <1 x i32> undef, i32 %24, i32 0, !dbg !1929
  %27 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %25, <1 x i32> %26, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true), !dbg !1929
  store <2 x i32> %27, <2 x i32>* %8, align 8, !dbg !1929, !tbaa !7
  %28 = load i32*, i32** %4, align 4, !dbg !1930, !tbaa !17
  %29 = load i32, i32* %28, align 4, !dbg !1930, !tbaa !13
  %30 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !1931, !tbaa !7
  %31 = insertelement <1 x i32> undef, i32 %29, i32 0, !dbg !1931
  %32 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %30, <1 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !1931
  store <2 x i32> %32, <2 x i32>* %9, align 8, !dbg !1931, !tbaa !7
  %33 = load i32*, i32** %4, align 4, !dbg !1932, !tbaa !17
  %34 = load i32, i32* %33, align 4, !dbg !1932, !tbaa !13
  %35 = add i32 %34, 4, !dbg !1933
  %36 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !1934, !tbaa !7
  %37 = insertelement <1 x i32> undef, i32 %35, i32 0, !dbg !1934
  %38 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %36, <1 x i32> %37, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true), !dbg !1934
  store <2 x i32> %38, <2 x i32>* %9, align 8, !dbg !1934, !tbaa !7
  %39 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !1935, !tbaa !7
  %40 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !1936, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi2EEvu2CMvbT0__ju2CMvbT0__T_(<2 x i32> %39, <2 x i32> %40), !dbg !1937
  %41 = load i32*, i32** %4, align 4, !dbg !1938, !tbaa !17
  %42 = load i32, i32* %41, align 4, !dbg !1939, !tbaa !13
  %43 = add i32 %42, 8, !dbg !1939
  store i32 %43, i32* %41, align 4, !dbg !1939, !tbaa !13
  ret void, !dbg !1940
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details18_cm_print_type_oclIxEENS_18SHADER_PRINTF_TYPEEv() #25 comdat {
  ret i32 6
}

; Function Attrs: nounwind readnone
declare <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32>, <1 x i32>, i32, i32, i32, i16, i32, i1) #2

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_Z20cm_svm_scatter_writeIjLi2EEvu2CMvbT0__ju2CMvbT0__T_(<2 x i32>, <2 x i32>) #6 comdat {
  %3 = alloca <2 x i32>, align 8
  %4 = alloca <2 x i32>, align 8
  store <2 x i32> %0, <2 x i32>* %3, align 8, !tbaa !7
  store <2 x i32> %1, <2 x i32>* %4, align 8, !tbaa !7
  %5 = load <2 x i32>, <2 x i32>* %3, align 8, !tbaa !7
  %6 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  call void @_ZN7details25cm_svm_scatter_write_implIjLi2EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_(<2 x i32> %5, <2 x i32> %6)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN7details25cm_svm_scatter_write_implIjLi2EEENSt9enable_ifIXclL_ZNS_L10isPowerOf2EjjET0_Li32EEEvE4typeEu2CMvbT0__ju2CMvbT0__T_(<2 x i32>, <2 x i32>) #8 comdat {
  %3 = alloca <2 x i32>, align 8
  %4 = alloca <2 x i32>, align 8
  %5 = alloca <2 x i64>, align 16
  store <2 x i32> %0, <2 x i32>* %3, align 8, !tbaa !7
  store <2 x i32> %1, <2 x i32>* %4, align 8, !tbaa !7
  %6 = load <2 x i32>, <2 x i32>* %3, align 8, !tbaa !7
  %7 = zext <2 x i32> %6 to <2 x i64>
  store <2 x i64> %7, <2 x i64>* %5, align 16, !tbaa !7
  %8 = load <2 x i64>, <2 x i64>* %5, align 16, !tbaa !7
  %9 = load <2 x i32>, <2 x i32>* %4, align 8, !tbaa !7
  call void @llvm.genx.svm.scatter.v2i1.v2i64.v2i32(<2 x i1> <i1 true, i1 true>, i32 0, <2 x i64> %8, <2 x i32> %9)
  ret void
}

declare dso_local void @_ZN7details37__cm_intrinsic_impl_svm_scatter_writeIjLi2ELi1EEEvu2CMvbT0__yu2CMvbmlT0_T1__T_(<2 x i64>, <2 x i32>) #24

; Function Attrs: nounwind
declare void @llvm.genx.svm.scatter.v2i1.v2i64.v2i32(<2 x i1>, i32, <2 x i64>, <2 x i32>) #27

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIyEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"*) #33 comdat align 2 !dbg !1941 {
  %2 = alloca %"struct.PrintInfo::Encoding4Uniform"*, align 4
  store %"struct.PrintInfo::Encoding4Uniform"* %0, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Uniform"*, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIyEENS_8EncodingEv(), !dbg !1942
  ret i8 %4, !dbg !1943
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter4callIyEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1944 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  call void @_ZN13UniformWriter8WriteArgIyEEvv(%class.UniformWriter* %3), !dbg !1945
  ret void, !dbg !1946
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter8WriteArgIyEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1947 {
  %2 = alloca %class.UniformWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %5 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %6 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1948
  store i32 %6, i32* %3, align 4, !dbg !1949, !tbaa !13
  %7 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1950
  store i32 %7, i32* %4, align 4, !dbg !1951, !tbaa !13
  %8 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %5, i32 0, i32 0, !dbg !1952
  %9 = load i32*, i32** %8, align 32, !dbg !1952, !tbaa !1774
  %10 = load i32, i32* %3, align 4, !dbg !1953, !tbaa !13
  %11 = load i32, i32* %4, align 4, !dbg !1954, !tbaa !13
  call void @_ZL24write_arg_with_promotionIyEvRjjj(i32* dereferenceable(4) %9, i32 %10, i32 %11), !dbg !1955
  ret void, !dbg !1956
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZL24write_arg_with_promotionIyEvRjjj(i32* dereferenceable(4), i32, i32) #6 !dbg !1957 {
  %4 = alloca i32*, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <2 x i32>, align 8
  %9 = alloca <2 x i32>, align 8
  store i32* %0, i32** %4, align 4, !tbaa !17
  store i32 %1, i32* %5, align 4, !tbaa !13
  store i32 %2, i32* %6, align 4, !tbaa !13
  %10 = call i32 @_ZN7details18_cm_print_type_oclIyEENS_18SHADER_PRINTF_TYPEEv(), !dbg !1958
  %11 = insertelement <1 x i32> undef, i32 %10, i32 0, !dbg !1958
  %12 = shufflevector <1 x i32> %11, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1958
  store <1 x i32> %12, <1 x i32>* %7, align 4, !dbg !1959, !tbaa !7
  %13 = load i32*, i32** %4, align 4, !dbg !1960, !tbaa !17
  %14 = load i32, i32* %13, align 4, !dbg !1960, !tbaa !13
  %15 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %14), !dbg !1961
  %16 = load <1 x i32>, <1 x i32>* %7, align 4, !dbg !1962, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %15, <1 x i32> %16), !dbg !1963
  %17 = load i32*, i32** %4, align 4, !dbg !1964, !tbaa !17
  %18 = load i32, i32* %17, align 4, !dbg !1965, !tbaa !13
  %19 = add i32 %18, 4, !dbg !1965
  store i32 %19, i32* %17, align 4, !dbg !1965, !tbaa !13
  %20 = load i32, i32* %5, align 4, !dbg !1966, !tbaa !13
  %21 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !1967, !tbaa !7
  %22 = insertelement <1 x i32> undef, i32 %20, i32 0, !dbg !1967
  %23 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %21, <1 x i32> %22, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !1967
  store <2 x i32> %23, <2 x i32>* %8, align 8, !dbg !1967, !tbaa !7
  %24 = load i32, i32* %6, align 4, !dbg !1968, !tbaa !13
  %25 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !1969, !tbaa !7
  %26 = insertelement <1 x i32> undef, i32 %24, i32 0, !dbg !1969
  %27 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %25, <1 x i32> %26, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true), !dbg !1969
  store <2 x i32> %27, <2 x i32>* %8, align 8, !dbg !1969, !tbaa !7
  %28 = load i32*, i32** %4, align 4, !dbg !1970, !tbaa !17
  %29 = load i32, i32* %28, align 4, !dbg !1970, !tbaa !13
  %30 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !1971, !tbaa !7
  %31 = insertelement <1 x i32> undef, i32 %29, i32 0, !dbg !1971
  %32 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %30, <1 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !1971
  store <2 x i32> %32, <2 x i32>* %9, align 8, !dbg !1971, !tbaa !7
  %33 = load i32*, i32** %4, align 4, !dbg !1972, !tbaa !17
  %34 = load i32, i32* %33, align 4, !dbg !1972, !tbaa !13
  %35 = add i32 %34, 4, !dbg !1973
  %36 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !1974, !tbaa !7
  %37 = insertelement <1 x i32> undef, i32 %35, i32 0, !dbg !1974
  %38 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %36, <1 x i32> %37, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true), !dbg !1974
  store <2 x i32> %38, <2 x i32>* %9, align 8, !dbg !1974, !tbaa !7
  %39 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !1975, !tbaa !7
  %40 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !1976, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi2EEvu2CMvbT0__ju2CMvbT0__T_(<2 x i32> %39, <2 x i32> %40), !dbg !1977
  %41 = load i32*, i32** %4, align 4, !dbg !1978, !tbaa !17
  %42 = load i32, i32* %41, align 4, !dbg !1979, !tbaa !13
  %43 = add i32 %42, 8, !dbg !1979
  store i32 %43, i32* %41, align 4, !dbg !1979, !tbaa !13
  ret void, !dbg !1980
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details18_cm_print_type_oclIyEENS_18SHADER_PRINTF_TYPEEv() #25 comdat {
  ret i32 6
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIdEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"*) #33 comdat align 2 !dbg !1981 {
  %2 = alloca %"struct.PrintInfo::Encoding4Uniform"*, align 4
  store %"struct.PrintInfo::Encoding4Uniform"* %0, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Uniform"*, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIdEENS_8EncodingEv(), !dbg !1982
  ret i8 %4, !dbg !1983
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter4callIdEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1984 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  call void @_ZN13UniformWriter8WriteArgIdEEvv(%class.UniformWriter* %3), !dbg !1985
  ret void, !dbg !1986
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter8WriteArgIdEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !1987 {
  %2 = alloca %class.UniformWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %5 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %6 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1988
  store i32 %6, i32* %3, align 4, !dbg !1989, !tbaa !13
  %7 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !1990
  store i32 %7, i32* %4, align 4, !dbg !1991, !tbaa !13
  %8 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %5, i32 0, i32 0, !dbg !1992
  %9 = load i32*, i32** %8, align 32, !dbg !1992, !tbaa !1774
  %10 = load i32, i32* %3, align 4, !dbg !1993, !tbaa !13
  %11 = load i32, i32* %4, align 4, !dbg !1994, !tbaa !13
  call void @_ZL24write_arg_with_promotionIdEvRjjj(i32* dereferenceable(4) %9, i32 %10, i32 %11), !dbg !1995
  ret void, !dbg !1996
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZL24write_arg_with_promotionIdEvRjjj(i32* dereferenceable(4), i32, i32) #6 !dbg !1997 {
  %4 = alloca i32*, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <2 x i32>, align 8
  %9 = alloca <2 x i32>, align 8
  store i32* %0, i32** %4, align 4, !tbaa !17
  store i32 %1, i32* %5, align 4, !tbaa !13
  store i32 %2, i32* %6, align 4, !tbaa !13
  %10 = call i32 @_ZN7details18_cm_print_type_oclIdEENS_18SHADER_PRINTF_TYPEEv(), !dbg !1998
  %11 = insertelement <1 x i32> undef, i32 %10, i32 0, !dbg !1998
  %12 = shufflevector <1 x i32> %11, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !1998
  store <1 x i32> %12, <1 x i32>* %7, align 4, !dbg !1999, !tbaa !7
  %13 = load i32*, i32** %4, align 4, !dbg !2000, !tbaa !17
  %14 = load i32, i32* %13, align 4, !dbg !2000, !tbaa !13
  %15 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %14), !dbg !2001
  %16 = load <1 x i32>, <1 x i32>* %7, align 4, !dbg !2002, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %15, <1 x i32> %16), !dbg !2003
  %17 = load i32*, i32** %4, align 4, !dbg !2004, !tbaa !17
  %18 = load i32, i32* %17, align 4, !dbg !2005, !tbaa !13
  %19 = add i32 %18, 4, !dbg !2005
  store i32 %19, i32* %17, align 4, !dbg !2005, !tbaa !13
  %20 = load i32, i32* %5, align 4, !dbg !2006, !tbaa !13
  %21 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !2007, !tbaa !7
  %22 = insertelement <1 x i32> undef, i32 %20, i32 0, !dbg !2007
  %23 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %21, <1 x i32> %22, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !2007
  store <2 x i32> %23, <2 x i32>* %8, align 8, !dbg !2007, !tbaa !7
  %24 = load i32, i32* %6, align 4, !dbg !2008, !tbaa !13
  %25 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !2009, !tbaa !7
  %26 = insertelement <1 x i32> undef, i32 %24, i32 0, !dbg !2009
  %27 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %25, <1 x i32> %26, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true), !dbg !2009
  store <2 x i32> %27, <2 x i32>* %8, align 8, !dbg !2009, !tbaa !7
  %28 = load i32*, i32** %4, align 4, !dbg !2010, !tbaa !17
  %29 = load i32, i32* %28, align 4, !dbg !2010, !tbaa !13
  %30 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !2011, !tbaa !7
  %31 = insertelement <1 x i32> undef, i32 %29, i32 0, !dbg !2011
  %32 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %30, <1 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef, i1 true), !dbg !2011
  store <2 x i32> %32, <2 x i32>* %9, align 8, !dbg !2011, !tbaa !7
  %33 = load i32*, i32** %4, align 4, !dbg !2012, !tbaa !17
  %34 = load i32, i32* %33, align 4, !dbg !2012, !tbaa !13
  %35 = add i32 %34, 4, !dbg !2013
  %36 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !2014, !tbaa !7
  %37 = insertelement <1 x i32> undef, i32 %35, i32 0, !dbg !2014
  %38 = call <2 x i32> @llvm.genx.wrregioni.v2i32.v1i32.i16.i1(<2 x i32> %36, <1 x i32> %37, i32 0, i32 1, i32 0, i16 4, i32 undef, i1 true), !dbg !2014
  store <2 x i32> %38, <2 x i32>* %9, align 8, !dbg !2014, !tbaa !7
  %39 = load <2 x i32>, <2 x i32>* %9, align 8, !dbg !2015, !tbaa !7
  %40 = load <2 x i32>, <2 x i32>* %8, align 8, !dbg !2016, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi2EEvu2CMvbT0__ju2CMvbT0__T_(<2 x i32> %39, <2 x i32> %40), !dbg !2017
  %41 = load i32*, i32** %4, align 4, !dbg !2018, !tbaa !17
  %42 = load i32, i32* %41, align 4, !dbg !2019, !tbaa !13
  %43 = add i32 %42, 8, !dbg !2019
  store i32 %43, i32* %41, align 4, !dbg !2019, !tbaa !13
  ret void, !dbg !2020
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details18_cm_print_type_oclIdEENS_18SHADER_PRINTF_TYPEEv() #25 comdat {
  ret i32 8
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Uniform4callIPvEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Uniform"*) #33 comdat align 2 !dbg !2021 {
  %2 = alloca %"struct.PrintInfo::Encoding4Uniform"*, align 4
  store %"struct.PrintInfo::Encoding4Uniform"* %0, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Uniform"*, %"struct.PrintInfo::Encoding4Uniform"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4UniformIPvEENS_8EncodingEv(), !dbg !2022
  ret i8 %4, !dbg !2023
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter4callIPvEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !2024 {
  %2 = alloca %class.UniformWriter*, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %3 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  call void @_ZN13UniformWriter8WriteArgIPvEEvv(%class.UniformWriter* %3), !dbg !2025
  ret void, !dbg !2026
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13UniformWriter8WriteArgIPvEEvv(%class.UniformWriter*) #25 comdat align 2 !dbg !2027 {
  %2 = alloca %class.UniformWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store %class.UniformWriter* %0, %class.UniformWriter** %2, align 4, !tbaa !17
  %5 = load %class.UniformWriter*, %class.UniformWriter** %2, align 4
  %6 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !2028
  store i32 %6, i32* %3, align 4, !dbg !2029, !tbaa !13
  %7 = call i32 @_ZN13UniformWriter16GetElementaryArgEv(%class.UniformWriter* %5), !dbg !2030
  store i32 %7, i32* %4, align 4, !dbg !2031, !tbaa !13
  %8 = getelementptr inbounds %class.UniformWriter, %class.UniformWriter* %5, i32 0, i32 0, !dbg !2032
  %9 = load i32*, i32** %8, align 32, !dbg !2032, !tbaa !1774
  %10 = load i32, i32* %3, align 4, !dbg !2033, !tbaa !13
  %11 = load i32, i32* %4, align 4, !dbg !2034, !tbaa !13
  call void @_ZL24write_arg_with_promotionIPvEvRjjj(i32* dereferenceable(4) %9, i32 %10, i32 %11), !dbg !2035
  ret void, !dbg !2036
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZL24write_arg_with_promotionIPvEvRjjj(i32* dereferenceable(4), i32, i32) #4 !dbg !2037 {
  %4 = alloca i32*, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca <1 x i32>, align 4
  %8 = alloca <1 x i32>, align 4
  store i32* %0, i32** %4, align 4, !tbaa !17
  store i32 %1, i32* %5, align 4, !tbaa !13
  store i32 %2, i32* %6, align 4, !tbaa !13
  %9 = call i32 @_ZN7details18_cm_print_type_oclIPvEENS_18SHADER_PRINTF_TYPEEv(), !dbg !2038
  %10 = insertelement <1 x i32> undef, i32 %9, i32 0, !dbg !2038
  %11 = shufflevector <1 x i32> %10, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !2038
  store <1 x i32> %11, <1 x i32>* %7, align 4, !dbg !2039, !tbaa !7
  %12 = load i32*, i32** %4, align 4, !dbg !2040, !tbaa !17
  %13 = load i32, i32* %12, align 4, !dbg !2040, !tbaa !13
  %14 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %13), !dbg !2041
  %15 = load <1 x i32>, <1 x i32>* %7, align 4, !dbg !2042, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %14, <1 x i32> %15), !dbg !2043
  %16 = load i32*, i32** %4, align 4, !dbg !2044, !tbaa !17
  %17 = load i32, i32* %16, align 4, !dbg !2045, !tbaa !13
  %18 = add i32 %17, 4, !dbg !2045
  store i32 %18, i32* %16, align 4, !dbg !2045, !tbaa !13
  %19 = load i32, i32* %5, align 4, !dbg !2046, !tbaa !13
  %20 = insertelement <1 x i32> undef, i32 %19, i32 0, !dbg !2046
  %21 = shufflevector <1 x i32> %20, <1 x i32> undef, <1 x i32> zeroinitializer, !dbg !2046
  store <1 x i32> %21, <1 x i32>* %8, align 4, !dbg !2047, !tbaa !7
  %22 = load i32*, i32** %4, align 4, !dbg !2048, !tbaa !17
  %23 = load i32, i32* %22, align 4, !dbg !2048, !tbaa !13
  %24 = call <1 x i32> @_ZL11vector_castIjEu2CMvb1_T_S0_(i32 %23), !dbg !2049
  %25 = load <1 x i32>, <1 x i32>* %8, align 4, !dbg !2050, !tbaa !7
  call void @_Z20cm_svm_scatter_writeIjLi1EEvu2CMvbT0__ju2CMvbT0__T_(<1 x i32> %24, <1 x i32> %25), !dbg !2051
  %26 = load i32*, i32** %4, align 4, !dbg !2052, !tbaa !17
  %27 = load i32, i32* %26, align 4, !dbg !2053, !tbaa !13
  %28 = add i32 %27, 4, !dbg !2053
  store i32 %28, i32* %26, align 4, !dbg !2053, !tbaa !13
  ret void, !dbg !2054
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal i32 @_ZN7details18_cm_print_type_oclIPvEENS_18SHADER_PRINTF_TYPEEv() #25 comdat {
  ret i32 7
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail14switchEncodingI13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET_T0_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2055 {
  %4 = alloca i8, align 1
  %5 = alloca %class.VaryingWriter, align 32
  %6 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  %7 = alloca %class.VaryingWriter, align 32
  %8 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  %9 = alloca %class.VaryingWriter, align 32
  %10 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  %11 = alloca %class.VaryingWriter, align 32
  %12 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  %13 = alloca %class.VaryingWriter, align 32
  %14 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  %15 = alloca %class.VaryingWriter, align 32
  %16 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  %17 = alloca %class.VaryingWriter, align 32
  %18 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  %19 = alloca %class.VaryingWriter, align 32
  %20 = alloca %"struct.PrintInfo::Encoding4Varying", align 1
  store i8 %0, i8* %4, align 1, !tbaa !193
  %21 = load i8, i8* %4, align 1, !dbg !2056, !tbaa !193
  %22 = bitcast %class.VaryingWriter* %5 to i8*, !dbg !2057
  %23 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !2057
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %22, i8* align 32 %23, i32 64, i1 false), !dbg !2057, !tbaa.struct !197
  %24 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIb13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext %21, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %5, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %6), !dbg !2058
  br i1 %24, label %60, label %25, !dbg !2059

; <label>:25:                                     ; preds = %3
  %26 = load i8, i8* %4, align 1, !dbg !2060, !tbaa !193
  %27 = bitcast %class.VaryingWriter* %7 to i8*, !dbg !2061
  %28 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !2061
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %27, i8* align 32 %28, i32 64, i1 false), !dbg !2061, !tbaa.struct !197
  %29 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIi13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext %26, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %7, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %8), !dbg !2062
  br i1 %29, label %60, label %30, !dbg !2063

; <label>:30:                                     ; preds = %25
  %31 = load i8, i8* %4, align 1, !dbg !2064, !tbaa !193
  %32 = bitcast %class.VaryingWriter* %9 to i8*, !dbg !2065
  %33 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !2065
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %32, i8* align 32 %33, i32 64, i1 false), !dbg !2065, !tbaa.struct !197
  %34 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIj13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext %31, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %9, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %10), !dbg !2066
  br i1 %34, label %60, label %35, !dbg !2067

; <label>:35:                                     ; preds = %30
  %36 = load i8, i8* %4, align 1, !dbg !2068, !tbaa !193
  %37 = bitcast %class.VaryingWriter* %11 to i8*, !dbg !2069
  %38 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !2069
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %37, i8* align 32 %38, i32 64, i1 false), !dbg !2069, !tbaa.struct !197
  %39 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIf13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext %36, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %11, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %12), !dbg !2070
  br i1 %39, label %60, label %40, !dbg !2071

; <label>:40:                                     ; preds = %35
  %41 = load i8, i8* %4, align 1, !dbg !2072, !tbaa !193
  %42 = bitcast %class.VaryingWriter* %13 to i8*, !dbg !2073
  %43 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !2073
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %42, i8* align 32 %43, i32 64, i1 false), !dbg !2073, !tbaa.struct !197
  %44 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIx13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext %41, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %13, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %14), !dbg !2074
  br i1 %44, label %60, label %45, !dbg !2075

; <label>:45:                                     ; preds = %40
  %46 = load i8, i8* %4, align 1, !dbg !2076, !tbaa !193
  %47 = bitcast %class.VaryingWriter* %15 to i8*, !dbg !2077
  %48 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !2077
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %47, i8* align 32 %48, i32 64, i1 false), !dbg !2077, !tbaa.struct !197
  %49 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIy13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext %46, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %15, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %16), !dbg !2078
  br i1 %49, label %60, label %50, !dbg !2079

; <label>:50:                                     ; preds = %45
  %51 = load i8, i8* %4, align 1, !dbg !2080, !tbaa !193
  %52 = bitcast %class.VaryingWriter* %17 to i8*, !dbg !2081
  %53 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !2081
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %52, i8* align 32 %53, i32 64, i1 false), !dbg !2081, !tbaa.struct !197
  %54 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingId13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext %51, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %17, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %18), !dbg !2082
  br i1 %54, label %60, label %55, !dbg !2083

; <label>:55:                                     ; preds = %50
  %56 = load i8, i8* %4, align 1, !dbg !2084, !tbaa !193
  %57 = bitcast %class.VaryingWriter* %19 to i8*, !dbg !2085
  %58 = bitcast %class.VaryingWriter* %1 to i8*, !dbg !2085
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 32 %57, i8* align 32 %58, i32 64, i1 false), !dbg !2085, !tbaa.struct !197
  %59 = call zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIPv13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext %56, %class.VaryingWriter* byval(%class.VaryingWriter) align 32 %19, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1 %20), !dbg !2086
  br label %60, !dbg !2083

; <label>:60:                                     ; preds = %55, %50, %45, %40, %35, %30, %25, %3
  %61 = phi i1 [ true, %50 ], [ true, %45 ], [ true, %40 ], [ true, %35 ], [ true, %30 ], [ true, %25 ], [ true, %3 ], [ %59, %55 ]
  ret i1 %61, !dbg !2087
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIb13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2088 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !2089, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !2089
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIbEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"* %2), !dbg !2090
  %9 = sext i8 %8 to i32, !dbg !2091
  %10 = icmp eq i32 %7, %9, !dbg !2092
  br i1 %10, label %11, label %12, !dbg !2089

; <label>:11:                                     ; preds = %3
  call void @_ZN13VaryingWriter4callIbEEvv(%class.VaryingWriter* %1), !dbg !2093
  store i1 true, i1* %4, align 1, !dbg !2094
  br label %13, !dbg !2094

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !2095
  br label %13, !dbg !2095

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !2096
  ret i1 %14, !dbg !2096
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIi13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2097 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !2098, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !2098
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIiEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"* %2), !dbg !2099
  %9 = sext i8 %8 to i32, !dbg !2100
  %10 = icmp eq i32 %7, %9, !dbg !2101
  br i1 %10, label %11, label %12, !dbg !2098

; <label>:11:                                     ; preds = %3
  call void @_ZN13VaryingWriter4callIiEEvv(%class.VaryingWriter* %1), !dbg !2102
  store i1 true, i1* %4, align 1, !dbg !2103
  br label %13, !dbg !2103

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !2104
  br label %13, !dbg !2104

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !2105
  ret i1 %14, !dbg !2105
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIj13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2106 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !2107, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !2107
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIjEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"* %2), !dbg !2108
  %9 = sext i8 %8 to i32, !dbg !2109
  %10 = icmp eq i32 %7, %9, !dbg !2110
  br i1 %10, label %11, label %12, !dbg !2107

; <label>:11:                                     ; preds = %3
  call void @_ZN13VaryingWriter4callIjEEvv(%class.VaryingWriter* %1), !dbg !2111
  store i1 true, i1* %4, align 1, !dbg !2112
  br label %13, !dbg !2112

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !2113
  br label %13, !dbg !2113

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !2114
  ret i1 %14, !dbg !2114
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIf13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2115 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !2116, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !2116
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIfEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"* %2), !dbg !2117
  %9 = sext i8 %8 to i32, !dbg !2118
  %10 = icmp eq i32 %7, %9, !dbg !2119
  br i1 %10, label %11, label %12, !dbg !2116

; <label>:11:                                     ; preds = %3
  call void @_ZN13VaryingWriter4callIfEEvv(%class.VaryingWriter* %1), !dbg !2120
  store i1 true, i1* %4, align 1, !dbg !2121
  br label %13, !dbg !2121

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !2122
  br label %13, !dbg !2122

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !2123
  ret i1 %14, !dbg !2123
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIx13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2124 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !2125, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !2125
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIxEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"* %2), !dbg !2126
  %9 = sext i8 %8 to i32, !dbg !2127
  %10 = icmp eq i32 %7, %9, !dbg !2128
  br i1 %10, label %11, label %12, !dbg !2125

; <label>:11:                                     ; preds = %3
  call void @_ZN13VaryingWriter4callIxEEvv(%class.VaryingWriter* %1), !dbg !2129
  store i1 true, i1* %4, align 1, !dbg !2130
  br label %13, !dbg !2130

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !2131
  br label %13, !dbg !2131

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !2132
  ret i1 %14, !dbg !2132
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIy13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2133 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !2134, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !2134
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIyEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"* %2), !dbg !2135
  %9 = sext i8 %8 to i32, !dbg !2136
  %10 = icmp eq i32 %7, %9, !dbg !2137
  br i1 %10, label %11, label %12, !dbg !2134

; <label>:11:                                     ; preds = %3
  call void @_ZN13VaryingWriter4callIyEEvv(%class.VaryingWriter* %1), !dbg !2138
  store i1 true, i1* %4, align 1, !dbg !2139
  br label %13, !dbg !2139

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !2140
  br label %13, !dbg !2140

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !2141
  ret i1 %14, !dbg !2141
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingId13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2142 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !2143, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !2143
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIdEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"* %2), !dbg !2144
  %9 = sext i8 %8 to i32, !dbg !2145
  %10 = icmp eq i32 %7, %9, !dbg !2146
  br i1 %10, label %11, label %12, !dbg !2143

; <label>:11:                                     ; preds = %3
  call void @_ZN13VaryingWriter4callIdEEvv(%class.VaryingWriter* %1), !dbg !2147
  store i1 true, i1* %4, align 1, !dbg !2148
  br label %13, !dbg !2148

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !2149
  br label %13, !dbg !2149

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !2150
  ret i1 %14, !dbg !2150
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal zeroext i1 @_ZN9PrintInfo6detail21applyIfProperEncodingIPv13VaryingWriterNS_16Encoding4VaryingEEEbNS_8EncodingET0_T1_(i8 signext, %class.VaryingWriter* byval(%class.VaryingWriter) align 32, %"struct.PrintInfo::Encoding4Varying"* byval(%"struct.PrintInfo::Encoding4Varying") align 1) #25 comdat !dbg !2151 {
  %4 = alloca i1, align 1
  %5 = alloca i8, align 1
  store i8 %0, i8* %5, align 1, !tbaa !193
  %6 = load i8, i8* %5, align 1, !dbg !2152, !tbaa !193
  %7 = sext i8 %6 to i32, !dbg !2152
  %8 = call signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIPvEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"* %2), !dbg !2153
  %9 = sext i8 %8 to i32, !dbg !2154
  %10 = icmp eq i32 %7, %9, !dbg !2155
  br i1 %10, label %11, label %12, !dbg !2152

; <label>:11:                                     ; preds = %3
  call void @_ZN13VaryingWriter4callIPvEEvv(%class.VaryingWriter* %1), !dbg !2156
  store i1 true, i1* %4, align 1, !dbg !2157
  br label %13, !dbg !2157

; <label>:12:                                     ; preds = %3
  store i1 false, i1* %4, align 1, !dbg !2158
  br label %13, !dbg !2158

; <label>:13:                                     ; preds = %12, %11
  %14 = load i1, i1* %4, align 1, !dbg !2159
  ret i1 %14, !dbg !2159
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIbEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"*) #33 comdat align 2 !dbg !2160 {
  %2 = alloca %"struct.PrintInfo::Encoding4Varying"*, align 4
  store %"struct.PrintInfo::Encoding4Varying"* %0, %"struct.PrintInfo::Encoding4Varying"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Varying"*, %"struct.PrintInfo::Encoding4Varying"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIbEENS_8EncodingEv(), !dbg !2161
  ret i8 %4, !dbg !2162
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter4callIbEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2163 {
  %2 = alloca %class.VaryingWriter*, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %3 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  call void @_ZN13VaryingWriter8WriteArgIbEEvv(%class.VaryingWriter* %3), !dbg !2164
  ret void, !dbg !2165
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter8WriteArgIbEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2166 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %6 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  store i32 0, i32* %3, align 4, !dbg !2167, !tbaa !13
  br label %7, !dbg !2168

; <label>:7:                                      ; preds = %45, %1
  %8 = load i32, i32* %3, align 4, !dbg !2169, !tbaa !13
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 2, !dbg !2170
  %10 = load i32, i32* %9, align 8, !dbg !2170, !tbaa !227
  %11 = icmp slt i32 %8, %10, !dbg !2171
  br i1 %11, label %12, label %48, !dbg !2172

; <label>:12:                                     ; preds = %7
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 3, !dbg !2173
  %14 = load i64, i64* %13, align 16, !dbg !2173, !tbaa !231
  %15 = load i32, i32* %3, align 4, !dbg !2174, !tbaa !13
  %16 = zext i32 %15 to i64, !dbg !2175
  %17 = shl i64 1, %16, !dbg !2175
  %18 = and i64 %14, %17, !dbg !2176
  %19 = icmp ne i64 %18, 0, !dbg !2173
  br i1 %19, label %20, label %29, !dbg !2173

; <label>:20:                                     ; preds = %12
  %21 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2177
  %22 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !2177, !tbaa !234
  %23 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %22, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2177
  %24 = extractelement <1 x i32> %23, i32 0, !dbg !2177
  store i32 %24, i32* %4, align 4, !dbg !2178, !tbaa !13
  %25 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2179
  %26 = load <5 x i32>, <5 x i32>* %25, align 32, !dbg !2179, !tbaa !234
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %26, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2179
  %28 = extractelement <1 x i32> %27, i32 0, !dbg !2179
  store i32 %28, i32* %5, align 4, !dbg !2180, !tbaa !13
  br label %38, !dbg !2181

; <label>:29:                                     ; preds = %12
  %30 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2182
  %31 = load <5 x i32>, <5 x i32>* %30, align 32, !dbg !2182, !tbaa !234
  %32 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef), !dbg !2182
  %33 = extractelement <1 x i32> %32, i32 0, !dbg !2182
  store i32 %33, i32* %4, align 4, !dbg !2183, !tbaa !13
  %34 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2184
  %35 = load <5 x i32>, <5 x i32>* %34, align 32, !dbg !2184, !tbaa !234
  %36 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %35, i32 0, i32 1, i32 0, i16 4, i32 undef), !dbg !2184
  %37 = extractelement <1 x i32> %36, i32 0, !dbg !2184
  store i32 %37, i32* %5, align 4, !dbg !2185, !tbaa !13
  br label %38

; <label>:38:                                     ; preds = %29, %20
  %39 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2186
  %40 = load i32*, i32** %39, align 32, !dbg !2186, !tbaa !2187
  %41 = load i32, i32* %4, align 4, !dbg !2188, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %40, i32 %41, i32 0), !dbg !2189
  call void @_ZN13VaryingWriter12WriteVecElemIbEEvv(%class.VaryingWriter* %6), !dbg !2190
  %42 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2191
  %43 = load i32*, i32** %42, align 32, !dbg !2191, !tbaa !2187
  %44 = load i32, i32* %5, align 4, !dbg !2192, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %43, i32 %44, i32 0), !dbg !2193
  br label %45, !dbg !2194

; <label>:45:                                     ; preds = %38
  %46 = load i32, i32* %3, align 4, !dbg !2195, !tbaa !13
  %47 = add nsw i32 %46, 1, !dbg !2195
  store i32 %47, i32* %3, align 4, !dbg !2195, !tbaa !13
  br label %7, !dbg !2172, !llvm.loop !2196

; <label>:48:                                     ; preds = %7
  ret void, !dbg !2197
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter12WriteVecElemIbEEvv(%class.VaryingWriter*) #28 comdat align 2 !dbg !2198 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca %class.UniformWriter, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %4 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  %5 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 0, !dbg !2199
  %6 = load i32*, i32** %5, align 32, !dbg !2199, !tbaa !2187
  %7 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 1, !dbg !2200
  %8 = load i32**, i32*** %7, align 4, !dbg !2200, !tbaa !2201
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 2, !dbg !2202
  %10 = load i32, i32* %9, align 8, !dbg !2202, !tbaa !227
  %11 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 3, !dbg !2203
  %12 = load i64, i64* %11, align 16, !dbg !2203, !tbaa !231
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 4, !dbg !2204
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !2204, !tbaa !234
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %3, i32* dereferenceable(4) %6, i32** dereferenceable(4) %8, i32 %10, i64 %12, <5 x i32> %14), !dbg !2205
  call void @_ZN13UniformWriter4callIbEEvv(%class.UniformWriter* %3), !dbg !2206
  ret void, !dbg !2207
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIiEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"*) #33 comdat align 2 !dbg !2208 {
  %2 = alloca %"struct.PrintInfo::Encoding4Varying"*, align 4
  store %"struct.PrintInfo::Encoding4Varying"* %0, %"struct.PrintInfo::Encoding4Varying"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Varying"*, %"struct.PrintInfo::Encoding4Varying"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIiEENS_8EncodingEv(), !dbg !2209
  ret i8 %4, !dbg !2210
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter4callIiEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2211 {
  %2 = alloca %class.VaryingWriter*, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %3 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  call void @_ZN13VaryingWriter8WriteArgIiEEvv(%class.VaryingWriter* %3), !dbg !2212
  ret void, !dbg !2213
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter8WriteArgIiEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2214 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %6 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  store i32 0, i32* %3, align 4, !dbg !2215, !tbaa !13
  br label %7, !dbg !2216

; <label>:7:                                      ; preds = %45, %1
  %8 = load i32, i32* %3, align 4, !dbg !2217, !tbaa !13
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 2, !dbg !2218
  %10 = load i32, i32* %9, align 8, !dbg !2218, !tbaa !227
  %11 = icmp slt i32 %8, %10, !dbg !2219
  br i1 %11, label %12, label %48, !dbg !2220

; <label>:12:                                     ; preds = %7
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 3, !dbg !2221
  %14 = load i64, i64* %13, align 16, !dbg !2221, !tbaa !231
  %15 = load i32, i32* %3, align 4, !dbg !2222, !tbaa !13
  %16 = zext i32 %15 to i64, !dbg !2223
  %17 = shl i64 1, %16, !dbg !2223
  %18 = and i64 %14, %17, !dbg !2224
  %19 = icmp ne i64 %18, 0, !dbg !2221
  br i1 %19, label %20, label %29, !dbg !2221

; <label>:20:                                     ; preds = %12
  %21 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2225
  %22 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !2225, !tbaa !234
  %23 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %22, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2225
  %24 = extractelement <1 x i32> %23, i32 0, !dbg !2225
  store i32 %24, i32* %4, align 4, !dbg !2226, !tbaa !13
  %25 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2227
  %26 = load <5 x i32>, <5 x i32>* %25, align 32, !dbg !2227, !tbaa !234
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %26, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2227
  %28 = extractelement <1 x i32> %27, i32 0, !dbg !2227
  store i32 %28, i32* %5, align 4, !dbg !2228, !tbaa !13
  br label %38, !dbg !2229

; <label>:29:                                     ; preds = %12
  %30 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2230
  %31 = load <5 x i32>, <5 x i32>* %30, align 32, !dbg !2230, !tbaa !234
  %32 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef), !dbg !2230
  %33 = extractelement <1 x i32> %32, i32 0, !dbg !2230
  store i32 %33, i32* %4, align 4, !dbg !2231, !tbaa !13
  %34 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2232
  %35 = load <5 x i32>, <5 x i32>* %34, align 32, !dbg !2232, !tbaa !234
  %36 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %35, i32 0, i32 1, i32 0, i16 4, i32 undef), !dbg !2232
  %37 = extractelement <1 x i32> %36, i32 0, !dbg !2232
  store i32 %37, i32* %5, align 4, !dbg !2233, !tbaa !13
  br label %38

; <label>:38:                                     ; preds = %29, %20
  %39 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2234
  %40 = load i32*, i32** %39, align 32, !dbg !2234, !tbaa !2187
  %41 = load i32, i32* %4, align 4, !dbg !2235, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %40, i32 %41, i32 0), !dbg !2236
  call void @_ZN13VaryingWriter12WriteVecElemIiEEvv(%class.VaryingWriter* %6), !dbg !2237
  %42 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2238
  %43 = load i32*, i32** %42, align 32, !dbg !2238, !tbaa !2187
  %44 = load i32, i32* %5, align 4, !dbg !2239, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %43, i32 %44, i32 0), !dbg !2240
  br label %45, !dbg !2241

; <label>:45:                                     ; preds = %38
  %46 = load i32, i32* %3, align 4, !dbg !2242, !tbaa !13
  %47 = add nsw i32 %46, 1, !dbg !2242
  store i32 %47, i32* %3, align 4, !dbg !2242, !tbaa !13
  br label %7, !dbg !2220, !llvm.loop !2243

; <label>:48:                                     ; preds = %7
  ret void, !dbg !2244
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter12WriteVecElemIiEEvv(%class.VaryingWriter*) #28 comdat align 2 !dbg !2245 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca %class.UniformWriter, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %4 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  %5 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 0, !dbg !2246
  %6 = load i32*, i32** %5, align 32, !dbg !2246, !tbaa !2187
  %7 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 1, !dbg !2247
  %8 = load i32**, i32*** %7, align 4, !dbg !2247, !tbaa !2201
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 2, !dbg !2248
  %10 = load i32, i32* %9, align 8, !dbg !2248, !tbaa !227
  %11 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 3, !dbg !2249
  %12 = load i64, i64* %11, align 16, !dbg !2249, !tbaa !231
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 4, !dbg !2250
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !2250, !tbaa !234
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %3, i32* dereferenceable(4) %6, i32** dereferenceable(4) %8, i32 %10, i64 %12, <5 x i32> %14), !dbg !2251
  call void @_ZN13UniformWriter4callIiEEvv(%class.UniformWriter* %3), !dbg !2252
  ret void, !dbg !2253
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIjEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"*) #33 comdat align 2 !dbg !2254 {
  %2 = alloca %"struct.PrintInfo::Encoding4Varying"*, align 4
  store %"struct.PrintInfo::Encoding4Varying"* %0, %"struct.PrintInfo::Encoding4Varying"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Varying"*, %"struct.PrintInfo::Encoding4Varying"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIjEENS_8EncodingEv(), !dbg !2255
  ret i8 %4, !dbg !2256
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter4callIjEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2257 {
  %2 = alloca %class.VaryingWriter*, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %3 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  call void @_ZN13VaryingWriter8WriteArgIjEEvv(%class.VaryingWriter* %3), !dbg !2258
  ret void, !dbg !2259
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter8WriteArgIjEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2260 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %6 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  store i32 0, i32* %3, align 4, !dbg !2261, !tbaa !13
  br label %7, !dbg !2262

; <label>:7:                                      ; preds = %45, %1
  %8 = load i32, i32* %3, align 4, !dbg !2263, !tbaa !13
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 2, !dbg !2264
  %10 = load i32, i32* %9, align 8, !dbg !2264, !tbaa !227
  %11 = icmp slt i32 %8, %10, !dbg !2265
  br i1 %11, label %12, label %48, !dbg !2266

; <label>:12:                                     ; preds = %7
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 3, !dbg !2267
  %14 = load i64, i64* %13, align 16, !dbg !2267, !tbaa !231
  %15 = load i32, i32* %3, align 4, !dbg !2268, !tbaa !13
  %16 = zext i32 %15 to i64, !dbg !2269
  %17 = shl i64 1, %16, !dbg !2269
  %18 = and i64 %14, %17, !dbg !2270
  %19 = icmp ne i64 %18, 0, !dbg !2267
  br i1 %19, label %20, label %29, !dbg !2267

; <label>:20:                                     ; preds = %12
  %21 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2271
  %22 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !2271, !tbaa !234
  %23 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %22, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2271
  %24 = extractelement <1 x i32> %23, i32 0, !dbg !2271
  store i32 %24, i32* %4, align 4, !dbg !2272, !tbaa !13
  %25 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2273
  %26 = load <5 x i32>, <5 x i32>* %25, align 32, !dbg !2273, !tbaa !234
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %26, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2273
  %28 = extractelement <1 x i32> %27, i32 0, !dbg !2273
  store i32 %28, i32* %5, align 4, !dbg !2274, !tbaa !13
  br label %38, !dbg !2275

; <label>:29:                                     ; preds = %12
  %30 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2276
  %31 = load <5 x i32>, <5 x i32>* %30, align 32, !dbg !2276, !tbaa !234
  %32 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef), !dbg !2276
  %33 = extractelement <1 x i32> %32, i32 0, !dbg !2276
  store i32 %33, i32* %4, align 4, !dbg !2277, !tbaa !13
  %34 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2278
  %35 = load <5 x i32>, <5 x i32>* %34, align 32, !dbg !2278, !tbaa !234
  %36 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %35, i32 0, i32 1, i32 0, i16 4, i32 undef), !dbg !2278
  %37 = extractelement <1 x i32> %36, i32 0, !dbg !2278
  store i32 %37, i32* %5, align 4, !dbg !2279, !tbaa !13
  br label %38

; <label>:38:                                     ; preds = %29, %20
  %39 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2280
  %40 = load i32*, i32** %39, align 32, !dbg !2280, !tbaa !2187
  %41 = load i32, i32* %4, align 4, !dbg !2281, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %40, i32 %41, i32 0), !dbg !2282
  call void @_ZN13VaryingWriter12WriteVecElemIjEEvv(%class.VaryingWriter* %6), !dbg !2283
  %42 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2284
  %43 = load i32*, i32** %42, align 32, !dbg !2284, !tbaa !2187
  %44 = load i32, i32* %5, align 4, !dbg !2285, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %43, i32 %44, i32 0), !dbg !2286
  br label %45, !dbg !2287

; <label>:45:                                     ; preds = %38
  %46 = load i32, i32* %3, align 4, !dbg !2288, !tbaa !13
  %47 = add nsw i32 %46, 1, !dbg !2288
  store i32 %47, i32* %3, align 4, !dbg !2288, !tbaa !13
  br label %7, !dbg !2266, !llvm.loop !2289

; <label>:48:                                     ; preds = %7
  ret void, !dbg !2290
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter12WriteVecElemIjEEvv(%class.VaryingWriter*) #28 comdat align 2 !dbg !2291 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca %class.UniformWriter, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %4 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  %5 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 0, !dbg !2292
  %6 = load i32*, i32** %5, align 32, !dbg !2292, !tbaa !2187
  %7 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 1, !dbg !2293
  %8 = load i32**, i32*** %7, align 4, !dbg !2293, !tbaa !2201
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 2, !dbg !2294
  %10 = load i32, i32* %9, align 8, !dbg !2294, !tbaa !227
  %11 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 3, !dbg !2295
  %12 = load i64, i64* %11, align 16, !dbg !2295, !tbaa !231
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 4, !dbg !2296
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !2296, !tbaa !234
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %3, i32* dereferenceable(4) %6, i32** dereferenceable(4) %8, i32 %10, i64 %12, <5 x i32> %14), !dbg !2297
  call void @_ZN13UniformWriter4callIjEEvv(%class.UniformWriter* %3), !dbg !2298
  ret void, !dbg !2299
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIfEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"*) #33 comdat align 2 !dbg !2300 {
  %2 = alloca %"struct.PrintInfo::Encoding4Varying"*, align 4
  store %"struct.PrintInfo::Encoding4Varying"* %0, %"struct.PrintInfo::Encoding4Varying"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Varying"*, %"struct.PrintInfo::Encoding4Varying"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIfEENS_8EncodingEv(), !dbg !2301
  ret i8 %4, !dbg !2302
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter4callIfEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2303 {
  %2 = alloca %class.VaryingWriter*, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %3 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  call void @_ZN13VaryingWriter8WriteArgIfEEvv(%class.VaryingWriter* %3), !dbg !2304
  ret void, !dbg !2305
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter8WriteArgIfEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2306 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %6 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  store i32 0, i32* %3, align 4, !dbg !2307, !tbaa !13
  br label %7, !dbg !2308

; <label>:7:                                      ; preds = %45, %1
  %8 = load i32, i32* %3, align 4, !dbg !2309, !tbaa !13
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 2, !dbg !2310
  %10 = load i32, i32* %9, align 8, !dbg !2310, !tbaa !227
  %11 = icmp slt i32 %8, %10, !dbg !2311
  br i1 %11, label %12, label %48, !dbg !2312

; <label>:12:                                     ; preds = %7
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 3, !dbg !2313
  %14 = load i64, i64* %13, align 16, !dbg !2313, !tbaa !231
  %15 = load i32, i32* %3, align 4, !dbg !2314, !tbaa !13
  %16 = zext i32 %15 to i64, !dbg !2315
  %17 = shl i64 1, %16, !dbg !2315
  %18 = and i64 %14, %17, !dbg !2316
  %19 = icmp ne i64 %18, 0, !dbg !2313
  br i1 %19, label %20, label %29, !dbg !2313

; <label>:20:                                     ; preds = %12
  %21 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2317
  %22 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !2317, !tbaa !234
  %23 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %22, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2317
  %24 = extractelement <1 x i32> %23, i32 0, !dbg !2317
  store i32 %24, i32* %4, align 4, !dbg !2318, !tbaa !13
  %25 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2319
  %26 = load <5 x i32>, <5 x i32>* %25, align 32, !dbg !2319, !tbaa !234
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %26, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2319
  %28 = extractelement <1 x i32> %27, i32 0, !dbg !2319
  store i32 %28, i32* %5, align 4, !dbg !2320, !tbaa !13
  br label %38, !dbg !2321

; <label>:29:                                     ; preds = %12
  %30 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2322
  %31 = load <5 x i32>, <5 x i32>* %30, align 32, !dbg !2322, !tbaa !234
  %32 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef), !dbg !2322
  %33 = extractelement <1 x i32> %32, i32 0, !dbg !2322
  store i32 %33, i32* %4, align 4, !dbg !2323, !tbaa !13
  %34 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2324
  %35 = load <5 x i32>, <5 x i32>* %34, align 32, !dbg !2324, !tbaa !234
  %36 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %35, i32 0, i32 1, i32 0, i16 4, i32 undef), !dbg !2324
  %37 = extractelement <1 x i32> %36, i32 0, !dbg !2324
  store i32 %37, i32* %5, align 4, !dbg !2325, !tbaa !13
  br label %38

; <label>:38:                                     ; preds = %29, %20
  %39 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2326
  %40 = load i32*, i32** %39, align 32, !dbg !2326, !tbaa !2187
  %41 = load i32, i32* %4, align 4, !dbg !2327, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %40, i32 %41, i32 0), !dbg !2328
  call void @_ZN13VaryingWriter12WriteVecElemIfEEvv(%class.VaryingWriter* %6), !dbg !2329
  %42 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2330
  %43 = load i32*, i32** %42, align 32, !dbg !2330, !tbaa !2187
  %44 = load i32, i32* %5, align 4, !dbg !2331, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %43, i32 %44, i32 0), !dbg !2332
  br label %45, !dbg !2333

; <label>:45:                                     ; preds = %38
  %46 = load i32, i32* %3, align 4, !dbg !2334, !tbaa !13
  %47 = add nsw i32 %46, 1, !dbg !2334
  store i32 %47, i32* %3, align 4, !dbg !2334, !tbaa !13
  br label %7, !dbg !2312, !llvm.loop !2335

; <label>:48:                                     ; preds = %7
  ret void, !dbg !2336
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter12WriteVecElemIfEEvv(%class.VaryingWriter*) #28 comdat align 2 !dbg !2337 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca %class.UniformWriter, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %4 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  %5 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 0, !dbg !2338
  %6 = load i32*, i32** %5, align 32, !dbg !2338, !tbaa !2187
  %7 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 1, !dbg !2339
  %8 = load i32**, i32*** %7, align 4, !dbg !2339, !tbaa !2201
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 2, !dbg !2340
  %10 = load i32, i32* %9, align 8, !dbg !2340, !tbaa !227
  %11 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 3, !dbg !2341
  %12 = load i64, i64* %11, align 16, !dbg !2341, !tbaa !231
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 4, !dbg !2342
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !2342, !tbaa !234
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %3, i32* dereferenceable(4) %6, i32** dereferenceable(4) %8, i32 %10, i64 %12, <5 x i32> %14), !dbg !2343
  call void @_ZN13UniformWriter4callIfEEvv(%class.UniformWriter* %3), !dbg !2344
  ret void, !dbg !2345
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIxEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"*) #33 comdat align 2 !dbg !2346 {
  %2 = alloca %"struct.PrintInfo::Encoding4Varying"*, align 4
  store %"struct.PrintInfo::Encoding4Varying"* %0, %"struct.PrintInfo::Encoding4Varying"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Varying"*, %"struct.PrintInfo::Encoding4Varying"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIxEENS_8EncodingEv(), !dbg !2347
  ret i8 %4, !dbg !2348
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter4callIxEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2349 {
  %2 = alloca %class.VaryingWriter*, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %3 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  call void @_ZN13VaryingWriter8WriteArgIxEEvv(%class.VaryingWriter* %3), !dbg !2350
  ret void, !dbg !2351
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter8WriteArgIxEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2352 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %6 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  store i32 0, i32* %3, align 4, !dbg !2353, !tbaa !13
  br label %7, !dbg !2354

; <label>:7:                                      ; preds = %45, %1
  %8 = load i32, i32* %3, align 4, !dbg !2355, !tbaa !13
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 2, !dbg !2356
  %10 = load i32, i32* %9, align 8, !dbg !2356, !tbaa !227
  %11 = icmp slt i32 %8, %10, !dbg !2357
  br i1 %11, label %12, label %48, !dbg !2358

; <label>:12:                                     ; preds = %7
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 3, !dbg !2359
  %14 = load i64, i64* %13, align 16, !dbg !2359, !tbaa !231
  %15 = load i32, i32* %3, align 4, !dbg !2360, !tbaa !13
  %16 = zext i32 %15 to i64, !dbg !2361
  %17 = shl i64 1, %16, !dbg !2361
  %18 = and i64 %14, %17, !dbg !2362
  %19 = icmp ne i64 %18, 0, !dbg !2359
  br i1 %19, label %20, label %29, !dbg !2359

; <label>:20:                                     ; preds = %12
  %21 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2363
  %22 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !2363, !tbaa !234
  %23 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %22, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2363
  %24 = extractelement <1 x i32> %23, i32 0, !dbg !2363
  store i32 %24, i32* %4, align 4, !dbg !2364, !tbaa !13
  %25 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2365
  %26 = load <5 x i32>, <5 x i32>* %25, align 32, !dbg !2365, !tbaa !234
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %26, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2365
  %28 = extractelement <1 x i32> %27, i32 0, !dbg !2365
  store i32 %28, i32* %5, align 4, !dbg !2366, !tbaa !13
  br label %38, !dbg !2367

; <label>:29:                                     ; preds = %12
  %30 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2368
  %31 = load <5 x i32>, <5 x i32>* %30, align 32, !dbg !2368, !tbaa !234
  %32 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef), !dbg !2368
  %33 = extractelement <1 x i32> %32, i32 0, !dbg !2368
  store i32 %33, i32* %4, align 4, !dbg !2369, !tbaa !13
  %34 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2370
  %35 = load <5 x i32>, <5 x i32>* %34, align 32, !dbg !2370, !tbaa !234
  %36 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %35, i32 0, i32 1, i32 0, i16 4, i32 undef), !dbg !2370
  %37 = extractelement <1 x i32> %36, i32 0, !dbg !2370
  store i32 %37, i32* %5, align 4, !dbg !2371, !tbaa !13
  br label %38

; <label>:38:                                     ; preds = %29, %20
  %39 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2372
  %40 = load i32*, i32** %39, align 32, !dbg !2372, !tbaa !2187
  %41 = load i32, i32* %4, align 4, !dbg !2373, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %40, i32 %41, i32 0), !dbg !2374
  call void @_ZN13VaryingWriter12WriteVecElemIxEEvv(%class.VaryingWriter* %6), !dbg !2375
  %42 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2376
  %43 = load i32*, i32** %42, align 32, !dbg !2376, !tbaa !2187
  %44 = load i32, i32* %5, align 4, !dbg !2377, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %43, i32 %44, i32 0), !dbg !2378
  br label %45, !dbg !2379

; <label>:45:                                     ; preds = %38
  %46 = load i32, i32* %3, align 4, !dbg !2380, !tbaa !13
  %47 = add nsw i32 %46, 1, !dbg !2380
  store i32 %47, i32* %3, align 4, !dbg !2380, !tbaa !13
  br label %7, !dbg !2358, !llvm.loop !2381

; <label>:48:                                     ; preds = %7
  ret void, !dbg !2382
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter12WriteVecElemIxEEvv(%class.VaryingWriter*) #28 comdat align 2 !dbg !2383 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca %class.UniformWriter, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %4 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  %5 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 0, !dbg !2384
  %6 = load i32*, i32** %5, align 32, !dbg !2384, !tbaa !2187
  %7 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 1, !dbg !2385
  %8 = load i32**, i32*** %7, align 4, !dbg !2385, !tbaa !2201
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 2, !dbg !2386
  %10 = load i32, i32* %9, align 8, !dbg !2386, !tbaa !227
  %11 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 3, !dbg !2387
  %12 = load i64, i64* %11, align 16, !dbg !2387, !tbaa !231
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 4, !dbg !2388
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !2388, !tbaa !234
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %3, i32* dereferenceable(4) %6, i32** dereferenceable(4) %8, i32 %10, i64 %12, <5 x i32> %14), !dbg !2389
  call void @_ZN13UniformWriter4callIxEEvv(%class.UniformWriter* %3), !dbg !2390
  ret void, !dbg !2391
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIyEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"*) #33 comdat align 2 !dbg !2392 {
  %2 = alloca %"struct.PrintInfo::Encoding4Varying"*, align 4
  store %"struct.PrintInfo::Encoding4Varying"* %0, %"struct.PrintInfo::Encoding4Varying"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Varying"*, %"struct.PrintInfo::Encoding4Varying"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIyEENS_8EncodingEv(), !dbg !2393
  ret i8 %4, !dbg !2394
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter4callIyEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2395 {
  %2 = alloca %class.VaryingWriter*, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %3 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  call void @_ZN13VaryingWriter8WriteArgIyEEvv(%class.VaryingWriter* %3), !dbg !2396
  ret void, !dbg !2397
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter8WriteArgIyEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2398 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %6 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  store i32 0, i32* %3, align 4, !dbg !2399, !tbaa !13
  br label %7, !dbg !2400

; <label>:7:                                      ; preds = %45, %1
  %8 = load i32, i32* %3, align 4, !dbg !2401, !tbaa !13
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 2, !dbg !2402
  %10 = load i32, i32* %9, align 8, !dbg !2402, !tbaa !227
  %11 = icmp slt i32 %8, %10, !dbg !2403
  br i1 %11, label %12, label %48, !dbg !2404

; <label>:12:                                     ; preds = %7
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 3, !dbg !2405
  %14 = load i64, i64* %13, align 16, !dbg !2405, !tbaa !231
  %15 = load i32, i32* %3, align 4, !dbg !2406, !tbaa !13
  %16 = zext i32 %15 to i64, !dbg !2407
  %17 = shl i64 1, %16, !dbg !2407
  %18 = and i64 %14, %17, !dbg !2408
  %19 = icmp ne i64 %18, 0, !dbg !2405
  br i1 %19, label %20, label %29, !dbg !2405

; <label>:20:                                     ; preds = %12
  %21 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2409
  %22 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !2409, !tbaa !234
  %23 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %22, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2409
  %24 = extractelement <1 x i32> %23, i32 0, !dbg !2409
  store i32 %24, i32* %4, align 4, !dbg !2410, !tbaa !13
  %25 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2411
  %26 = load <5 x i32>, <5 x i32>* %25, align 32, !dbg !2411, !tbaa !234
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %26, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2411
  %28 = extractelement <1 x i32> %27, i32 0, !dbg !2411
  store i32 %28, i32* %5, align 4, !dbg !2412, !tbaa !13
  br label %38, !dbg !2413

; <label>:29:                                     ; preds = %12
  %30 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2414
  %31 = load <5 x i32>, <5 x i32>* %30, align 32, !dbg !2414, !tbaa !234
  %32 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef), !dbg !2414
  %33 = extractelement <1 x i32> %32, i32 0, !dbg !2414
  store i32 %33, i32* %4, align 4, !dbg !2415, !tbaa !13
  %34 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2416
  %35 = load <5 x i32>, <5 x i32>* %34, align 32, !dbg !2416, !tbaa !234
  %36 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %35, i32 0, i32 1, i32 0, i16 4, i32 undef), !dbg !2416
  %37 = extractelement <1 x i32> %36, i32 0, !dbg !2416
  store i32 %37, i32* %5, align 4, !dbg !2417, !tbaa !13
  br label %38

; <label>:38:                                     ; preds = %29, %20
  %39 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2418
  %40 = load i32*, i32** %39, align 32, !dbg !2418, !tbaa !2187
  %41 = load i32, i32* %4, align 4, !dbg !2419, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %40, i32 %41, i32 0), !dbg !2420
  call void @_ZN13VaryingWriter12WriteVecElemIyEEvv(%class.VaryingWriter* %6), !dbg !2421
  %42 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2422
  %43 = load i32*, i32** %42, align 32, !dbg !2422, !tbaa !2187
  %44 = load i32, i32* %5, align 4, !dbg !2423, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %43, i32 %44, i32 0), !dbg !2424
  br label %45, !dbg !2425

; <label>:45:                                     ; preds = %38
  %46 = load i32, i32* %3, align 4, !dbg !2426, !tbaa !13
  %47 = add nsw i32 %46, 1, !dbg !2426
  store i32 %47, i32* %3, align 4, !dbg !2426, !tbaa !13
  br label %7, !dbg !2404, !llvm.loop !2427

; <label>:48:                                     ; preds = %7
  ret void, !dbg !2428
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter12WriteVecElemIyEEvv(%class.VaryingWriter*) #28 comdat align 2 !dbg !2429 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca %class.UniformWriter, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %4 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  %5 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 0, !dbg !2430
  %6 = load i32*, i32** %5, align 32, !dbg !2430, !tbaa !2187
  %7 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 1, !dbg !2431
  %8 = load i32**, i32*** %7, align 4, !dbg !2431, !tbaa !2201
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 2, !dbg !2432
  %10 = load i32, i32* %9, align 8, !dbg !2432, !tbaa !227
  %11 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 3, !dbg !2433
  %12 = load i64, i64* %11, align 16, !dbg !2433, !tbaa !231
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 4, !dbg !2434
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !2434, !tbaa !234
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %3, i32* dereferenceable(4) %6, i32** dereferenceable(4) %8, i32 %10, i64 %12, <5 x i32> %14), !dbg !2435
  call void @_ZN13UniformWriter4callIyEEvv(%class.UniformWriter* %3), !dbg !2436
  ret void, !dbg !2437
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIdEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"*) #33 comdat align 2 !dbg !2438 {
  %2 = alloca %"struct.PrintInfo::Encoding4Varying"*, align 4
  store %"struct.PrintInfo::Encoding4Varying"* %0, %"struct.PrintInfo::Encoding4Varying"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Varying"*, %"struct.PrintInfo::Encoding4Varying"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIdEENS_8EncodingEv(), !dbg !2439
  ret i8 %4, !dbg !2440
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter4callIdEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2441 {
  %2 = alloca %class.VaryingWriter*, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %3 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  call void @_ZN13VaryingWriter8WriteArgIdEEvv(%class.VaryingWriter* %3), !dbg !2442
  ret void, !dbg !2443
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter8WriteArgIdEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2444 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %6 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  store i32 0, i32* %3, align 4, !dbg !2445, !tbaa !13
  br label %7, !dbg !2446

; <label>:7:                                      ; preds = %45, %1
  %8 = load i32, i32* %3, align 4, !dbg !2447, !tbaa !13
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 2, !dbg !2448
  %10 = load i32, i32* %9, align 8, !dbg !2448, !tbaa !227
  %11 = icmp slt i32 %8, %10, !dbg !2449
  br i1 %11, label %12, label %48, !dbg !2450

; <label>:12:                                     ; preds = %7
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 3, !dbg !2451
  %14 = load i64, i64* %13, align 16, !dbg !2451, !tbaa !231
  %15 = load i32, i32* %3, align 4, !dbg !2452, !tbaa !13
  %16 = zext i32 %15 to i64, !dbg !2453
  %17 = shl i64 1, %16, !dbg !2453
  %18 = and i64 %14, %17, !dbg !2454
  %19 = icmp ne i64 %18, 0, !dbg !2451
  br i1 %19, label %20, label %29, !dbg !2451

; <label>:20:                                     ; preds = %12
  %21 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2455
  %22 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !2455, !tbaa !234
  %23 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %22, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2455
  %24 = extractelement <1 x i32> %23, i32 0, !dbg !2455
  store i32 %24, i32* %4, align 4, !dbg !2456, !tbaa !13
  %25 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2457
  %26 = load <5 x i32>, <5 x i32>* %25, align 32, !dbg !2457, !tbaa !234
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %26, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2457
  %28 = extractelement <1 x i32> %27, i32 0, !dbg !2457
  store i32 %28, i32* %5, align 4, !dbg !2458, !tbaa !13
  br label %38, !dbg !2459

; <label>:29:                                     ; preds = %12
  %30 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2460
  %31 = load <5 x i32>, <5 x i32>* %30, align 32, !dbg !2460, !tbaa !234
  %32 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef), !dbg !2460
  %33 = extractelement <1 x i32> %32, i32 0, !dbg !2460
  store i32 %33, i32* %4, align 4, !dbg !2461, !tbaa !13
  %34 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2462
  %35 = load <5 x i32>, <5 x i32>* %34, align 32, !dbg !2462, !tbaa !234
  %36 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %35, i32 0, i32 1, i32 0, i16 4, i32 undef), !dbg !2462
  %37 = extractelement <1 x i32> %36, i32 0, !dbg !2462
  store i32 %37, i32* %5, align 4, !dbg !2463, !tbaa !13
  br label %38

; <label>:38:                                     ; preds = %29, %20
  %39 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2464
  %40 = load i32*, i32** %39, align 32, !dbg !2464, !tbaa !2187
  %41 = load i32, i32* %4, align 4, !dbg !2465, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %40, i32 %41, i32 0), !dbg !2466
  call void @_ZN13VaryingWriter12WriteVecElemIdEEvv(%class.VaryingWriter* %6), !dbg !2467
  %42 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2468
  %43 = load i32*, i32** %42, align 32, !dbg !2468, !tbaa !2187
  %44 = load i32, i32* %5, align 4, !dbg !2469, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %43, i32 %44, i32 0), !dbg !2470
  br label %45, !dbg !2471

; <label>:45:                                     ; preds = %38
  %46 = load i32, i32* %3, align 4, !dbg !2472, !tbaa !13
  %47 = add nsw i32 %46, 1, !dbg !2472
  store i32 %47, i32* %3, align 4, !dbg !2472, !tbaa !13
  br label %7, !dbg !2450, !llvm.loop !2473

; <label>:48:                                     ; preds = %7
  ret void, !dbg !2474
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter12WriteVecElemIdEEvv(%class.VaryingWriter*) #28 comdat align 2 !dbg !2475 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca %class.UniformWriter, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %4 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  %5 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 0, !dbg !2476
  %6 = load i32*, i32** %5, align 32, !dbg !2476, !tbaa !2187
  %7 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 1, !dbg !2477
  %8 = load i32**, i32*** %7, align 4, !dbg !2477, !tbaa !2201
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 2, !dbg !2478
  %10 = load i32, i32* %9, align 8, !dbg !2478, !tbaa !227
  %11 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 3, !dbg !2479
  %12 = load i64, i64* %11, align 16, !dbg !2479, !tbaa !231
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 4, !dbg !2480
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !2480, !tbaa !234
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %3, i32* dereferenceable(4) %6, i32** dereferenceable(4) %8, i32 %10, i64 %12, <5 x i32> %14), !dbg !2481
  call void @_ZN13UniformWriter4callIdEEvv(%class.UniformWriter* %3), !dbg !2482
  ret void, !dbg !2483
}

; Function Attrs: alwaysinline nounwind
define internal signext i8 @_ZNK9PrintInfo16Encoding4Varying4callIPvEENS_8EncodingEv(%"struct.PrintInfo::Encoding4Varying"*) #33 comdat align 2 !dbg !2484 {
  %2 = alloca %"struct.PrintInfo::Encoding4Varying"*, align 4
  store %"struct.PrintInfo::Encoding4Varying"* %0, %"struct.PrintInfo::Encoding4Varying"** %2, align 4, !tbaa !17
  %3 = load %"struct.PrintInfo::Encoding4Varying"*, %"struct.PrintInfo::Encoding4Varying"** %2, align 4
  %4 = call signext i8 @_ZN9PrintInfo19getEncoding4VaryingIPvEENS_8EncodingEv(), !dbg !2485
  ret i8 %4, !dbg !2486
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter4callIPvEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2487 {
  %2 = alloca %class.VaryingWriter*, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %3 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  call void @_ZN13VaryingWriter8WriteArgIPvEEvv(%class.VaryingWriter* %3), !dbg !2488
  ret void, !dbg !2489
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter8WriteArgIPvEEvv(%class.VaryingWriter*) #25 comdat align 2 !dbg !2490 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %6 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  store i32 0, i32* %3, align 4, !dbg !2491, !tbaa !13
  br label %7, !dbg !2492

; <label>:7:                                      ; preds = %45, %1
  %8 = load i32, i32* %3, align 4, !dbg !2493, !tbaa !13
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 2, !dbg !2494
  %10 = load i32, i32* %9, align 8, !dbg !2494, !tbaa !227
  %11 = icmp slt i32 %8, %10, !dbg !2495
  br i1 %11, label %12, label %48, !dbg !2496

; <label>:12:                                     ; preds = %7
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 3, !dbg !2497
  %14 = load i64, i64* %13, align 16, !dbg !2497, !tbaa !231
  %15 = load i32, i32* %3, align 4, !dbg !2498, !tbaa !13
  %16 = zext i32 %15 to i64, !dbg !2499
  %17 = shl i64 1, %16, !dbg !2499
  %18 = and i64 %14, %17, !dbg !2500
  %19 = icmp ne i64 %18, 0, !dbg !2497
  br i1 %19, label %20, label %29, !dbg !2497

; <label>:20:                                     ; preds = %12
  %21 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2501
  %22 = load <5 x i32>, <5 x i32>* %21, align 32, !dbg !2501, !tbaa !234
  %23 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %22, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2501
  %24 = extractelement <1 x i32> %23, i32 0, !dbg !2501
  store i32 %24, i32* %4, align 4, !dbg !2502, !tbaa !13
  %25 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2503
  %26 = load <5 x i32>, <5 x i32>* %25, align 32, !dbg !2503, !tbaa !234
  %27 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %26, i32 0, i32 1, i32 0, i16 8, i32 undef), !dbg !2503
  %28 = extractelement <1 x i32> %27, i32 0, !dbg !2503
  store i32 %28, i32* %5, align 4, !dbg !2504, !tbaa !13
  br label %38, !dbg !2505

; <label>:29:                                     ; preds = %12
  %30 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2506
  %31 = load <5 x i32>, <5 x i32>* %30, align 32, !dbg !2506, !tbaa !234
  %32 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %31, i32 0, i32 1, i32 0, i16 0, i32 undef), !dbg !2506
  %33 = extractelement <1 x i32> %32, i32 0, !dbg !2506
  store i32 %33, i32* %4, align 4, !dbg !2507, !tbaa !13
  %34 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 4, !dbg !2508
  %35 = load <5 x i32>, <5 x i32>* %34, align 32, !dbg !2508, !tbaa !234
  %36 = call <1 x i32> @llvm.genx.rdregioni.v1i32.v5i32.i16(<5 x i32> %35, i32 0, i32 1, i32 0, i16 4, i32 undef), !dbg !2508
  %37 = extractelement <1 x i32> %36, i32 0, !dbg !2508
  store i32 %37, i32* %5, align 4, !dbg !2509, !tbaa !13
  br label %38

; <label>:38:                                     ; preds = %29, %20
  %39 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2510
  %40 = load i32*, i32** %39, align 32, !dbg !2510, !tbaa !2187
  %41 = load i32, i32* %4, align 4, !dbg !2511, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %40, i32 %41, i32 0), !dbg !2512
  call void @_ZN13VaryingWriter12WriteVecElemIPvEEvv(%class.VaryingWriter* %6), !dbg !2513
  %42 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %6, i32 0, i32 0, !dbg !2514
  %43 = load i32*, i32** %42, align 32, !dbg !2514, !tbaa !2187
  %44 = load i32, i32* %5, align 4, !dbg !2515, !tbaa !13
  call void @_ZL24write_arg_with_promotionIRA1_KcEvRjjj(i32* dereferenceable(4) %43, i32 %44, i32 0), !dbg !2516
  br label %45, !dbg !2517

; <label>:45:                                     ; preds = %38
  %46 = load i32, i32* %3, align 4, !dbg !2518, !tbaa !13
  %47 = add nsw i32 %46, 1, !dbg !2518
  store i32 %47, i32* %3, align 4, !dbg !2518, !tbaa !13
  br label %7, !dbg !2496, !llvm.loop !2519

; <label>:48:                                     ; preds = %7
  ret void, !dbg !2520
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @_ZN13VaryingWriter12WriteVecElemIPvEEvv(%class.VaryingWriter*) #28 comdat align 2 !dbg !2521 {
  %2 = alloca %class.VaryingWriter*, align 4
  %3 = alloca %class.UniformWriter, align 32
  store %class.VaryingWriter* %0, %class.VaryingWriter** %2, align 4, !tbaa !17
  %4 = load %class.VaryingWriter*, %class.VaryingWriter** %2, align 4
  %5 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 0, !dbg !2522
  %6 = load i32*, i32** %5, align 32, !dbg !2522, !tbaa !2187
  %7 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 1, !dbg !2523
  %8 = load i32**, i32*** %7, align 4, !dbg !2523, !tbaa !2201
  %9 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 2, !dbg !2524
  %10 = load i32, i32* %9, align 8, !dbg !2524, !tbaa !227
  %11 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 3, !dbg !2525
  %12 = load i64, i64* %11, align 16, !dbg !2525, !tbaa !231
  %13 = getelementptr inbounds %class.VaryingWriter, %class.VaryingWriter* %4, i32 0, i32 4, !dbg !2526
  %14 = load <5 x i32>, <5 x i32>* %13, align 32, !dbg !2526, !tbaa !234
  call void @_ZN13UniformWriterC2ERjRPKjiyu2CMvb5_i(%class.UniformWriter* %3, i32* dereferenceable(4) %6, i32** dereferenceable(4) %8, i32 %10, i64 %12, <5 x i32> %14), !dbg !2527
  call void @_ZN13UniformWriter4callIPvEEvv(%class.UniformWriter* %3), !dbg !2528
  ret void, !dbg !2529
}

attributes #0 = { noinline nounwind "CMBuiltin" "CMFloatControl"="0" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="8" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="16" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { noinline nounwind "CMBuiltin" "CMFloatControl"="0" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="16" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="32" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline nounwind "CMBuiltin" "CMFloatControl"="0" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="32" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="64" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noinline nounwind "CMBuiltin" "CMFloatControl"="0" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="64" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { noinline nounwind "CMBuiltin" "CMFloatControl"="0" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="256" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { noinline nounwind "CMBuiltin" "CMFloatControl"="0" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="256" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { noinline nounwind "CMBuiltin" "CMFloatControl"="0" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #14 = { noinline nounwind "CMBuiltin" "CMFloatControl"="48" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="32" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #15 = { noinline nounwind "CMBuiltin" "CMFloatControl"="48" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="64" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #16 = { noinline nounwind "CMBuiltin" "CMFloatControl"="48" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #17 = { noinline nounwind "CMBuiltin" "CMFloatControl"="48" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="256" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #18 = { noinline nounwind "CMBuiltin" "CMFloatControl"="48" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #19 = { noinline nounwind "CMBuiltin" "CMFloatControl"="48" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="1024" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #20 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="1024" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #21 = { noinline nounwind "CMBuiltin" "CMFloatControl"="48" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="8" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #22 = { noinline nounwind "CMBuiltin" "CMFloatControl"="48" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="16" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #23 = { noinline nounwind "CMBuiltin" "CMGenxNoSIMDPred" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="1024" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #24 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #25 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #26 = { noinline nounwind "CMBuiltin" "CMGenxNoSIMDPred" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="160" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #27 = { nounwind }
attributes #28 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="160" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #29 = { noinline nounwind "CMBuiltin" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #30 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #31 = { alwaysinline inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="800" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #32 = { argmemonly nounwind }
attributes #33 = { alwaysinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!genx.kernels = !{}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.1 (ssh://gerrit-gfx.intel.com:29418/gfx/cmc/clang 61d84c8cc7a1840fc8c03b9b141db309011ab4c3) (ssh://gerrit-gfx.intel.com:29418/gfx/cmc/llvm 7add6781ee5118530a30e8c8de1be942a2139139)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "builtins/builtins-c-genx.cpp", directory: "/home/ispc")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.1 (ssh://gerrit-gfx.intel.com:29418/gfx/cmc/clang 61d84c8cc7a1840fc8c03b9b141db309011ab4c3) (ssh://gerrit-gfx.intel.com:29418/gfx/cmc/llvm 7add6781ee5118530a30e8c8de1be942a2139139)"}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{i32 6940}
!11 = !{!12, !12, i64 0}
!12 = !{!"short", !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !8, i64 0}
!15 = distinct !DISubprogram(name: "__do_print_cm", scope: !1, file: !1, line: 270, type: !16, scopeLine: 274, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !2)
!17 = !{!18, !18, i64 0}
!18 = !{!"any pointer", !8, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"long long", !8, i64 0}
!21 = !DILocation(line: 281, column: 41, scope: !15)
!22 = !DILocation(line: 281, column: 61, scope: !15)
!23 = !DILocation(line: 281, column: 81, scope: !15)
!24 = !DILocation(line: 281, column: 79, scope: !15)
!25 = !DILocation(line: 281, column: 59, scope: !15)
!26 = !DILocation(line: 281, column: 88, scope: !15)
!27 = !DILocation(line: 281, column: 38, scope: !15)
!28 = !DILocation(line: 280, column: 22, scope: !15)
!29 = !DILocation(line: 282, column: 24, scope: !15)
!30 = !DILocation(line: 282, column: 18, scope: !15)
!31 = !{!32, !32, i64 0}
!32 = !{!"SurfaceIndex", !8, i64 0}
!33 = !DILocation(line: 283, column: 55, scope: !15)
!34 = !DILocation(line: 283, column: 60, scope: !15)
!35 = !DILocation(line: 283, column: 24, scope: !15)
!36 = !DILocation(line: 283, column: 10, scope: !15)
!37 = !DILocation(line: 284, column: 23, scope: !15)
!38 = !DILocation(line: 284, column: 35, scope: !15)
!39 = !DILocation(line: 284, column: 50, scope: !15)
!40 = !DILocation(line: 284, column: 10, scope: !15)
!41 = !DILocation(line: 286, column: 25, scope: !15)
!42 = !DILocation(line: 286, column: 37, scope: !15)
!43 = !DILocation(line: 286, column: 43, scope: !15)
!44 = !DILocation(line: 286, column: 50, scope: !15)
!45 = !DILocation(line: 286, column: 15, scope: !15)
!46 = !DILocation(line: 287, column: 62, scope: !15)
!47 = !DILocation(line: 287, column: 70, scope: !15)
!48 = !DILocation(line: 287, column: 47, scope: !15)
!49 = !DILocation(line: 287, column: 32, scope: !15)
!50 = !DILocation(line: 288, column: 31, scope: !15)
!51 = !DILocation(line: 288, column: 36, scope: !15)
!52 = !DILocation(line: 288, column: 49, scope: !15)
!53 = !DILocation(line: 288, column: 5, scope: !15)
!54 = !DILocation(line: 289, column: 1, scope: !15)
!55 = !{i32 6925}
!56 = distinct !DISubprogram(name: "ArgWriter", scope: !1, file: !1, line: 127, type: !16, scopeLine: 128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!57 = !DILocation(line: 128, column: 11, scope: !56)
!58 = !DILocation(line: 128, column: 19, scope: !56)
!59 = !{!60, !14, i64 0}
!60 = !{!"_ZTS9ArgWriter", !14, i64 0, !18, i64 4, !14, i64 8, !14, i64 12, !20, i64 16}
!61 = !DILocation(line: 128, column: 28, scope: !56)
!62 = !DILocation(line: 128, column: 34, scope: !56)
!63 = !{!60, !18, i64 4}
!64 = !DILocation(line: 128, column: 41, scope: !56)
!65 = !{!60, !14, i64 8}
!66 = !DILocation(line: 128, column: 56, scope: !56)
!67 = !DILocation(line: 128, column: 63, scope: !56)
!68 = !{!60, !14, i64 12}
!69 = !DILocation(line: 128, column: 71, scope: !56)
!70 = !DILocation(line: 128, column: 77, scope: !56)
!71 = !{!60, !20, i64 16}
!72 = !DILocation(line: 128, column: 84, scope: !56)
!73 = distinct !DISubprogram(name: "GetFormatedStr<ArgWriter>", scope: !74, file: !74, line: 159, type: !16, scopeLine: 159, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!74 = !DIFile(filename: "./builtins/builtins-c-common.hpp", directory: "/home/ispc")
!75 = !DILocation(line: 161, column: 9, scope: !73)
!76 = !DILocation(line: 162, column: 9, scope: !73)
!77 = !DILocation(line: 164, column: 5, scope: !73)
!78 = !DILocation(line: 165, column: 41, scope: !73)
!79 = !DILocation(line: 165, column: 63, scope: !73)
!80 = !DILocation(line: 165, column: 80, scope: !73)
!81 = !DILocation(line: 165, column: 27, scope: !73)
!82 = !DILocation(line: 165, column: 14, scope: !73)
!83 = !DILocation(line: 166, column: 19, scope: !73)
!84 = !DILocation(line: 166, column: 16, scope: !73)
!85 = !DILocation(line: 167, column: 28, scope: !73)
!86 = !DILocation(line: 167, column: 25, scope: !73)
!87 = !DILocation(line: 168, column: 22, scope: !73)
!88 = !DILocation(line: 168, column: 19, scope: !73)
!89 = !DILocation(line: 169, column: 14, scope: !73)
!90 = !DILocation(line: 169, column: 24, scope: !73)
!91 = !DILocation(line: 169, column: 28, scope: !73)
!92 = !DILocation(line: 169, column: 27, scope: !73)
!93 = !DILocation(line: 169, column: 35, scope: !73)
!94 = !DILocation(line: 169, column: 13, scope: !73)
!95 = !DILocation(line: 171, column: 13, scope: !73)
!96 = !DILocation(line: 173, column: 63, scope: !73)
!97 = !DILocation(line: 173, column: 62, scope: !73)
!98 = !DILocation(line: 173, column: 70, scope: !73)
!99 = !DILocation(line: 173, column: 45, scope: !73)
!100 = !DILocation(line: 173, column: 36, scope: !73)
!101 = !DILocation(line: 174, column: 60, scope: !73)
!102 = !DILocation(line: 174, column: 77, scope: !73)
!103 = !DILocation(line: 174, column: 22, scope: !73)
!104 = !DILocation(line: 174, column: 20, scope: !73)
!105 = !DILocation(line: 175, column: 28, scope: !73)
!106 = !DILocation(line: 175, column: 25, scope: !73)
!107 = !DILocation(line: 176, column: 22, scope: !73)
!108 = !DILocation(line: 176, column: 19, scope: !73)
!109 = !DILocation(line: 177, column: 5, scope: !73)
!110 = !DILocation(line: 164, column: 13, scope: !73)
!111 = !DILocation(line: 164, column: 23, scope: !73)
!112 = distinct !{!112, !77, !109}
!113 = !DILocation(line: 178, column: 18, scope: !73)
!114 = !DILocation(line: 178, column: 35, scope: !73)
!115 = !DILocation(line: 179, column: 12, scope: !73)
!116 = !DILocation(line: 179, column: 5, scope: !73)
!117 = distinct !DISubprogram(name: "__do_print_lz", scope: !1, file: !1, line: 454, type: !16, scopeLine: 456, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!118 = !DILocation(line: 458, column: 23, scope: !117)
!119 = !DILocation(line: 458, column: 14, scope: !117)
!120 = !DILocation(line: 460, column: 41, scope: !117)
!121 = !DILocation(line: 460, column: 62, scope: !117)
!122 = !DILocation(line: 460, column: 60, scope: !117)
!123 = !DILocation(line: 460, column: 70, scope: !117)
!124 = !DILocation(line: 460, column: 68, scope: !117)
!125 = !DILocation(line: 460, column: 56, scope: !117)
!126 = !DILocation(line: 460, column: 86, scope: !117)
!127 = !DILocation(line: 460, column: 38, scope: !117)
!128 = !DILocation(line: 461, column: 23, scope: !117)
!129 = !DILocation(line: 461, column: 45, scope: !117)
!130 = !DILocation(line: 461, column: 53, scope: !117)
!131 = !DILocation(line: 461, column: 51, scope: !117)
!132 = !DILocation(line: 461, column: 43, scope: !117)
!133 = !DILocation(line: 461, column: 74, scope: !117)
!134 = !DILocation(line: 460, column: 104, scope: !117)
!135 = !DILocation(line: 460, column: 10, scope: !117)
!136 = !DILocation(line: 464, column: 50, scope: !117)
!137 = !DILocation(line: 464, column: 58, scope: !117)
!138 = !DILocation(line: 464, column: 15, scope: !117)
!139 = !DILocation(line: 464, column: 12, scope: !117)
!140 = !DILocation(line: 467, column: 38, scope: !117)
!141 = !DILocation(line: 467, column: 26, scope: !117)
!142 = !DILocation(line: 467, column: 59, scope: !117)
!143 = !DILocation(line: 467, column: 47, scope: !117)
!144 = !DILocation(line: 467, column: 5, scope: !117)
!145 = !DILocation(line: 468, column: 12, scope: !117)
!146 = !DILocation(line: 470, column: 19, scope: !117)
!147 = !DILocation(line: 470, column: 10, scope: !117)
!148 = !DILocation(line: 472, column: 5, scope: !117)
!149 = !DILocation(line: 472, column: 13, scope: !117)
!150 = !DILocation(line: 472, column: 12, scope: !117)
!151 = !DILocation(line: 472, column: 19, scope: !117)
!152 = !DILocation(line: 473, column: 69, scope: !117)
!153 = !DILocation(line: 473, column: 68, scope: !117)
!154 = !DILocation(line: 474, column: 63, scope: !117)
!155 = !DILocation(line: 474, column: 70, scope: !117)
!156 = !DILocation(line: 474, column: 76, scope: !117)
!157 = !DILocation(line: 474, column: 35, scope: !117)
!158 = !DILocation(line: 475, column: 63, scope: !117)
!159 = !DILocation(line: 475, column: 70, scope: !117)
!160 = !DILocation(line: 475, column: 76, scope: !117)
!161 = !DILocation(line: 475, column: 35, scope: !117)
!162 = !DILocation(line: 473, column: 9, scope: !117)
!163 = !DILocation(line: 476, column: 5, scope: !117)
!164 = !DILocation(line: 472, column: 28, scope: !117)
!165 = distinct !{!165, !148, !163}
!166 = !DILocation(line: 477, column: 1, scope: !117)
!167 = !{i32 6926}
!168 = !{!169, !169, i64 0}
!169 = !{!"_ZTS14CmAtomicOpType", !8, i64 0}
!170 = distinct !DISubprogram(name: "vector_cast<unsigned int>", scope: !1, file: !1, line: 296, type: !16, scopeLine: 296, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!171 = !DILocation(line: 296, column: 97, scope: !170)
!172 = !DILocation(line: 296, column: 82, scope: !170)
!173 = !DILocation(line: 296, column: 75, scope: !170)
!174 = distinct !DISubprogram(name: "vector_cast<int>", scope: !1, file: !1, line: 296, type: !16, scopeLine: 296, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!175 = !DILocation(line: 296, column: 97, scope: !174)
!176 = !DILocation(line: 296, column: 82, scope: !174)
!177 = !DILocation(line: 296, column: 75, scope: !174)
!178 = distinct !DISubprogram(name: "get_auxiliary_str", scope: !1, file: !1, line: 302, type: !16, scopeLine: 302, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!179 = !DILocation(line: 304, column: 26, scope: !178)
!180 = !DILocation(line: 304, column: 24, scope: !178)
!181 = !DILocation(line: 305, column: 27, scope: !178)
!182 = !DILocation(line: 305, column: 25, scope: !178)
!183 = !DILocation(line: 306, column: 21, scope: !178)
!184 = !DILocation(line: 306, column: 19, scope: !178)
!185 = !DILocation(line: 307, column: 21, scope: !178)
!186 = !DILocation(line: 307, column: 19, scope: !178)
!187 = !DILocation(line: 308, column: 20, scope: !178)
!188 = !DILocation(line: 308, column: 18, scope: !178)
!189 = !DILocation(line: 309, column: 12, scope: !178)
!190 = !DILocation(line: 309, column: 5, scope: !178)
!191 = distinct !DISubprogram(name: "switchEncoding<UniformWriter, VaryingWriter>", scope: !192, file: !192, line: 167, type: !16, scopeLine: 167, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!192 = !DIFile(filename: "./src/builtins-info.h", directory: "/home/ispc")
!193 = !{!194, !194, i64 0}
!194 = !{!"_ZTSN9PrintInfo8EncodingE", !8, i64 0}
!195 = !DILocation(line: 168, column: 35, scope: !191)
!196 = !DILocation(line: 168, column: 41, scope: !191)
!197 = !{i64 0, i64 4, !17, i64 4, i64 4, !17, i64 8, i64 4, !13, i64 16, i64 8, !19, i64 32, i64 32, !7}
!198 = !DILocation(line: 168, column: 12, scope: !191)
!199 = !DILocation(line: 168, column: 47, scope: !191)
!200 = !DILocation(line: 168, column: 73, scope: !191)
!201 = !DILocation(line: 168, column: 79, scope: !191)
!202 = !DILocation(line: 168, column: 50, scope: !191)
!203 = !DILocation(line: 168, column: 5, scope: !191)
!204 = distinct !DISubprogram(name: "UniformWriter", scope: !1, file: !1, line: 348, type: !16, scopeLine: 350, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!205 = !DILocation(line: 350, column: 11, scope: !204)
!206 = !DILocation(line: 350, column: 18, scope: !204)
!207 = !DILocation(line: 350, column: 29, scope: !204)
!208 = !DILocation(line: 350, column: 34, scope: !204)
!209 = !DILocation(line: 350, column: 43, scope: !204)
!210 = !DILocation(line: 350, column: 49, scope: !204)
!211 = !{!212, !14, i64 8}
!212 = !{!"_ZTS13UniformWriter", !18, i64 0, !18, i64 4, !14, i64 8, !20, i64 16, !8, i64 32}
!213 = !DILocation(line: 350, column: 59, scope: !204)
!214 = !DILocation(line: 350, column: 64, scope: !204)
!215 = !{!212, !20, i64 16}
!216 = !DILocation(line: 350, column: 73, scope: !204)
!217 = !DILocation(line: 350, column: 80, scope: !204)
!218 = !{!212, !8, i64 32}
!219 = !DILocation(line: 350, column: 91, scope: !204)
!220 = distinct !DISubprogram(name: "VaryingWriter", scope: !1, file: !1, line: 390, type: !16, scopeLine: 392, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!221 = !DILocation(line: 392, column: 11, scope: !220)
!222 = !DILocation(line: 392, column: 18, scope: !220)
!223 = !DILocation(line: 392, column: 29, scope: !220)
!224 = !DILocation(line: 392, column: 34, scope: !220)
!225 = !DILocation(line: 392, column: 43, scope: !220)
!226 = !DILocation(line: 392, column: 49, scope: !220)
!227 = !{!228, !14, i64 8}
!228 = !{!"_ZTS13VaryingWriter", !18, i64 0, !18, i64 4, !14, i64 8, !20, i64 16, !8, i64 32}
!229 = !DILocation(line: 392, column: 59, scope: !220)
!230 = !DILocation(line: 392, column: 64, scope: !220)
!231 = !{!228, !20, i64 16}
!232 = !DILocation(line: 392, column: 73, scope: !220)
!233 = !DILocation(line: 392, column: 80, scope: !220)
!234 = !{!228, !8, i64 32}
!235 = !DILocation(line: 392, column: 91, scope: !220)
!236 = distinct !DISubprogram(name: "__num_cores", scope: !1, file: !1, line: 480, type: !16, scopeLine: 480, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!237 = !DILocation(line: 480, column: 55, scope: !236)
!238 = !{i32 7134}
!239 = distinct !DISubprogram(name: "check", scope: !240, file: !240, line: 65, type: !16, scopeLine: 65, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!240 = !DIFile(filename: "cmc/install/include/cm/cm_atomic.h", directory: "/home")
!241 = !DILocation(line: 65, column: 1, scope: !239)
!242 = !{i32 7127}
!243 = !{i32 6838}
!244 = !{i32 7128}
!245 = !{i32 7007}
!246 = !{i32 6927}
!247 = !{i32 6939}
!248 = distinct !DISubprogram(name: "CopyPlainText<128>", scope: !74, file: !74, line: 134, type: !16, scopeLine: 134, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!249 = !DILocation(line: 135, column: 44, scope: !248)
!250 = !DILocation(line: 135, column: 57, scope: !248)
!251 = !DILocation(line: 135, column: 65, scope: !248)
!252 = !DILocation(line: 135, column: 12, scope: !248)
!253 = !DILocation(line: 135, column: 5, scope: !248)
!254 = distinct !DISubprogram(name: "Arg2Str<ArgWriter>", scope: !74, file: !74, line: 110, type: !16, scopeLine: 110, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!255 = !DILocation(line: 112, column: 29, scope: !254)
!256 = !DILocation(line: 112, column: 35, scope: !254)
!257 = !DILocation(line: 112, column: 5, scope: !254)
!258 = !DILocation(line: 112, column: 51, scope: !254)
!259 = !DILocation(line: 112, column: 77, scope: !254)
!260 = !DILocation(line: 112, column: 83, scope: !254)
!261 = !DILocation(line: 112, column: 54, scope: !254)
!262 = !DILocation(line: 112, column: 99, scope: !254)
!263 = !DILocation(line: 113, column: 37, scope: !254)
!264 = !DILocation(line: 113, column: 43, scope: !254)
!265 = !DILocation(line: 113, column: 9, scope: !254)
!266 = !DILocation(line: 113, column: 59, scope: !254)
!267 = !DILocation(line: 113, column: 87, scope: !254)
!268 = !DILocation(line: 113, column: 93, scope: !254)
!269 = !DILocation(line: 113, column: 62, scope: !254)
!270 = !DILocation(line: 113, column: 109, scope: !254)
!271 = !DILocation(line: 114, column: 38, scope: !254)
!272 = !DILocation(line: 114, column: 44, scope: !254)
!273 = !DILocation(line: 114, column: 9, scope: !254)
!274 = !DILocation(line: 114, column: 60, scope: !254)
!275 = !DILocation(line: 115, column: 47, scope: !254)
!276 = !DILocation(line: 115, column: 53, scope: !254)
!277 = !DILocation(line: 115, column: 9, scope: !254)
!278 = !DILocation(line: 115, column: 69, scope: !254)
!279 = !DILocation(line: 116, column: 35, scope: !254)
!280 = !DILocation(line: 116, column: 41, scope: !254)
!281 = !DILocation(line: 116, column: 9, scope: !254)
!282 = !DILocation(line: 116, column: 57, scope: !254)
!283 = !DILocation(line: 116, column: 86, scope: !254)
!284 = !DILocation(line: 116, column: 92, scope: !254)
!285 = !DILocation(line: 116, column: 60, scope: !254)
!286 = !DILocation(line: 117, column: 12, scope: !254)
!287 = !DILocation(line: 117, column: 5, scope: !254)
!288 = distinct !DISubprogram(name: "CopyFullText<100, 128>", scope: !74, file: !74, line: 148, type: !16, scopeLine: 149, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!289 = !DILocation(line: 150, column: 44, scope: !288)
!290 = !DILocation(line: 150, column: 57, scope: !288)
!291 = !DILocation(line: 150, column: 65, scope: !288)
!292 = !DILocation(line: 150, column: 12, scope: !288)
!293 = !DILocation(line: 150, column: 5, scope: !288)
!294 = distinct !DISubprogram(name: "CopyTillSep<'%', '\5Cx00', const char *, 128>", scope: !74, file: !74, line: 57, type: !16, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!295 = !DILocation(line: 58, column: 29, scope: !294)
!296 = !DILocation(line: 58, column: 16, scope: !294)
!297 = !DILocation(line: 59, column: 5, scope: !294)
!298 = !DILocation(line: 59, column: 14, scope: !294)
!299 = !DILocation(line: 59, column: 18, scope: !294)
!300 = !DILocation(line: 59, column: 26, scope: !294)
!301 = !DILocation(line: 59, column: 38, scope: !294)
!302 = !DILocation(line: 59, column: 43, scope: !294)
!303 = !DILocation(line: 59, column: 46, scope: !294)
!304 = !DILocation(line: 0, scope: !294)
!305 = !DILocation(line: 60, column: 25, scope: !294)
!306 = !DILocation(line: 60, column: 35, scope: !294)
!307 = !DILocation(line: 60, column: 19, scope: !294)
!308 = !DILocation(line: 60, column: 13, scope: !294)
!309 = !DILocation(line: 60, column: 23, scope: !294)
!310 = !DILocation(line: 61, column: 9, scope: !294)
!311 = distinct !{!311, !297, !312}
!312 = !DILocation(line: 62, column: 5, scope: !294)
!313 = !DILocation(line: 63, column: 12, scope: !294)
!314 = !DILocation(line: 63, column: 21, scope: !294)
!315 = !DILocation(line: 63, column: 19, scope: !294)
!316 = !DILocation(line: 63, column: 5, scope: !294)
!317 = distinct !DISubprogram(name: "Arg2StrIfSuitable<bool, ArgWriter>", scope: !74, file: !74, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!318 = !DILocation(line: 104, column: 36, scope: !317)
!319 = !DILocation(line: 104, column: 42, scope: !317)
!320 = !DILocation(line: 104, column: 12, scope: !317)
!321 = !DILocation(line: 104, column: 58, scope: !317)
!322 = !DILocation(line: 104, column: 85, scope: !317)
!323 = !DILocation(line: 104, column: 91, scope: !317)
!324 = !DILocation(line: 104, column: 61, scope: !317)
!325 = !DILocation(line: 104, column: 5, scope: !317)
!326 = distinct !DISubprogram(name: "Arg2StrIfSuitable<int, ArgWriter>", scope: !74, file: !74, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!327 = !DILocation(line: 104, column: 36, scope: !326)
!328 = !DILocation(line: 104, column: 42, scope: !326)
!329 = !DILocation(line: 104, column: 12, scope: !326)
!330 = !DILocation(line: 104, column: 58, scope: !326)
!331 = !DILocation(line: 104, column: 85, scope: !326)
!332 = !DILocation(line: 104, column: 91, scope: !326)
!333 = !DILocation(line: 104, column: 61, scope: !326)
!334 = !DILocation(line: 104, column: 5, scope: !326)
!335 = distinct !DISubprogram(name: "Arg2StrIfSuitable<unsigned int, ArgWriter>", scope: !74, file: !74, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!336 = !DILocation(line: 104, column: 36, scope: !335)
!337 = !DILocation(line: 104, column: 42, scope: !335)
!338 = !DILocation(line: 104, column: 12, scope: !335)
!339 = !DILocation(line: 104, column: 58, scope: !335)
!340 = !DILocation(line: 104, column: 85, scope: !335)
!341 = !DILocation(line: 104, column: 91, scope: !335)
!342 = !DILocation(line: 104, column: 61, scope: !335)
!343 = !DILocation(line: 104, column: 5, scope: !335)
!344 = distinct !DISubprogram(name: "Arg2StrIfSuitable<float, ArgWriter>", scope: !74, file: !74, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!345 = !DILocation(line: 104, column: 36, scope: !344)
!346 = !DILocation(line: 104, column: 42, scope: !344)
!347 = !DILocation(line: 104, column: 12, scope: !344)
!348 = !DILocation(line: 104, column: 58, scope: !344)
!349 = !DILocation(line: 104, column: 85, scope: !344)
!350 = !DILocation(line: 104, column: 91, scope: !344)
!351 = !DILocation(line: 104, column: 61, scope: !344)
!352 = !DILocation(line: 104, column: 5, scope: !344)
!353 = distinct !DISubprogram(name: "Arg2StrIfSuitable<long long, ArgWriter>", scope: !74, file: !74, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!354 = !DILocation(line: 104, column: 36, scope: !353)
!355 = !DILocation(line: 104, column: 42, scope: !353)
!356 = !DILocation(line: 104, column: 12, scope: !353)
!357 = !DILocation(line: 104, column: 58, scope: !353)
!358 = !DILocation(line: 104, column: 85, scope: !353)
!359 = !DILocation(line: 104, column: 91, scope: !353)
!360 = !DILocation(line: 104, column: 61, scope: !353)
!361 = !DILocation(line: 104, column: 5, scope: !353)
!362 = distinct !DISubprogram(name: "Arg2StrIfSuitable<unsigned long long, ArgWriter>", scope: !74, file: !74, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!363 = !DILocation(line: 104, column: 36, scope: !362)
!364 = !DILocation(line: 104, column: 42, scope: !362)
!365 = !DILocation(line: 104, column: 12, scope: !362)
!366 = !DILocation(line: 104, column: 58, scope: !362)
!367 = !DILocation(line: 104, column: 85, scope: !362)
!368 = !DILocation(line: 104, column: 91, scope: !362)
!369 = !DILocation(line: 104, column: 61, scope: !362)
!370 = !DILocation(line: 104, column: 5, scope: !362)
!371 = distinct !DISubprogram(name: "Arg2StrIfSuitable<double, ArgWriter>", scope: !74, file: !74, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!372 = !DILocation(line: 104, column: 36, scope: !371)
!373 = !DILocation(line: 104, column: 42, scope: !371)
!374 = !DILocation(line: 104, column: 12, scope: !371)
!375 = !DILocation(line: 104, column: 58, scope: !371)
!376 = !DILocation(line: 104, column: 85, scope: !371)
!377 = !DILocation(line: 104, column: 91, scope: !371)
!378 = !DILocation(line: 104, column: 61, scope: !371)
!379 = !DILocation(line: 104, column: 5, scope: !371)
!380 = distinct !DISubprogram(name: "Arg2StrIfSuitable<void *, ArgWriter>", scope: !74, file: !74, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!381 = !DILocation(line: 104, column: 36, scope: !380)
!382 = !DILocation(line: 104, column: 42, scope: !380)
!383 = !DILocation(line: 104, column: 12, scope: !380)
!384 = !DILocation(line: 104, column: 58, scope: !380)
!385 = !DILocation(line: 104, column: 85, scope: !380)
!386 = !DILocation(line: 104, column: 91, scope: !380)
!387 = !DILocation(line: 104, column: 61, scope: !380)
!388 = !DILocation(line: 104, column: 5, scope: !380)
!389 = distinct !DISubprogram(name: "UniArg2StrIfSuitable<bool, ArgWriter>", scope: !74, file: !74, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!390 = !DILocation(line: 82, column: 9, scope: !389)
!391 = !DILocation(line: 82, column: 17, scope: !389)
!392 = !DILocation(line: 82, column: 14, scope: !389)
!393 = !DILocation(line: 83, column: 15, scope: !389)
!394 = !DILocation(line: 83, column: 34, scope: !389)
!395 = !DILocation(line: 83, column: 13, scope: !389)
!396 = !DILocation(line: 84, column: 9, scope: !389)
!397 = !DILocation(line: 86, column: 5, scope: !389)
!398 = !DILocation(line: 87, column: 1, scope: !389)
!399 = distinct !DISubprogram(name: "VarArg2StrIfSuitable<bool, ArgWriter>", scope: !74, file: !74, line: 92, type: !16, scopeLine: 92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!400 = !DILocation(line: 93, column: 9, scope: !399)
!401 = !DILocation(line: 93, column: 17, scope: !399)
!402 = !DILocation(line: 93, column: 14, scope: !399)
!403 = !DILocation(line: 94, column: 15, scope: !399)
!404 = !DILocation(line: 94, column: 34, scope: !399)
!405 = !DILocation(line: 94, column: 13, scope: !399)
!406 = !DILocation(line: 95, column: 9, scope: !399)
!407 = !DILocation(line: 97, column: 5, scope: !399)
!408 = !DILocation(line: 98, column: 1, scope: !399)
!409 = distinct !DISubprogram(name: "getEncoding4Uniform<bool>", scope: !192, file: !192, line: 75, type: !16, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!410 = !DILocation(line: 75, column: 69, scope: !409)
!411 = distinct !DISubprogram(name: "uniform2Str<bool>", scope: !1, file: !1, line: 140, type: !16, scopeLine: 140, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!412 = !DILocation(line: 142, column: 59, scope: !411)
!413 = !DILocation(line: 142, column: 40, scope: !411)
!414 = !DILocation(line: 142, column: 27, scope: !411)
!415 = !DILocation(line: 142, column: 14, scope: !411)
!416 = !DILocation(line: 143, column: 13, scope: !411)
!417 = !DILocation(line: 143, column: 25, scope: !411)
!418 = !DILocation(line: 144, column: 16, scope: !411)
!419 = !DILocation(line: 144, column: 9, scope: !411)
!420 = distinct !DISubprogram(name: "CopyFullText<100>", scope: !74, file: !74, line: 141, type: !16, scopeLine: 141, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!421 = !DILocation(line: 142, column: 39, scope: !420)
!422 = !DILocation(line: 142, column: 52, scope: !420)
!423 = !DILocation(line: 142, column: 60, scope: !420)
!424 = !DILocation(line: 142, column: 12, scope: !420)
!425 = !DILocation(line: 142, column: 5, scope: !420)
!426 = distinct !DISubprogram(name: "ValueAdapter<bool>", scope: !74, file: !74, line: 129, type: !16, scopeLine: 129, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!427 = !{!428, !428, i64 0}
!428 = !{!"bool", !8, i64 0}
!429 = !DILocation(line: 129, column: 90, scope: !426)
!430 = !{i8 0, i8 2}
!431 = !DILocation(line: 129, column: 64, scope: !426)
!432 = !DILocation(line: 129, column: 57, scope: !426)
!433 = distinct !DISubprogram(name: "GetElementaryArg", scope: !1, file: !1, line: 177, type: !16, scopeLine: 177, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!434 = !DILocation(line: 177, column: 45, scope: !433)
!435 = !DILocation(line: 177, column: 51, scope: !433)
!436 = !DILocation(line: 177, column: 61, scope: !433)
!437 = !DILocation(line: 177, column: 38, scope: !433)
!438 = distinct !DISubprogram(name: "CopyTillSep<'\5Cx00', const char *, 100>", scope: !74, file: !74, line: 57, type: !16, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!439 = !DILocation(line: 58, column: 29, scope: !438)
!440 = !DILocation(line: 58, column: 16, scope: !438)
!441 = !DILocation(line: 59, column: 5, scope: !438)
!442 = !DILocation(line: 59, column: 14, scope: !438)
!443 = !DILocation(line: 59, column: 18, scope: !438)
!444 = !DILocation(line: 59, column: 26, scope: !438)
!445 = !DILocation(line: 59, column: 43, scope: !438)
!446 = !DILocation(line: 59, column: 46, scope: !438)
!447 = !DILocation(line: 0, scope: !438)
!448 = !DILocation(line: 60, column: 25, scope: !438)
!449 = !DILocation(line: 60, column: 35, scope: !438)
!450 = !DILocation(line: 60, column: 19, scope: !438)
!451 = !DILocation(line: 60, column: 13, scope: !438)
!452 = !DILocation(line: 60, column: 23, scope: !438)
!453 = !DILocation(line: 61, column: 9, scope: !438)
!454 = distinct !{!454, !441, !455}
!455 = !DILocation(line: 62, column: 5, scope: !438)
!456 = !DILocation(line: 63, column: 12, scope: !438)
!457 = !DILocation(line: 63, column: 21, scope: !438)
!458 = !DILocation(line: 63, column: 19, scope: !438)
!459 = !DILocation(line: 63, column: 5, scope: !438)
!460 = distinct !DISubprogram(name: "ValueAdapterImpl", scope: !74, file: !74, line: 51, type: !16, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!461 = !DILocation(line: 51, column: 63, scope: !460)
!462 = !DILocation(line: 51, column: 56, scope: !460)
!463 = distinct !DISubprogram(name: "getEncoding4Varying<bool>", scope: !192, file: !192, line: 84, type: !16, scopeLine: 84, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!464 = !DILocation(line: 84, column: 69, scope: !463)
!465 = distinct !DISubprogram(name: "varying2Str<bool>", scope: !1, file: !1, line: 147, type: !16, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!466 = !DILocation(line: 149, column: 16, scope: !465)
!467 = !DILocation(line: 150, column: 13, scope: !465)
!468 = !DILocation(line: 151, column: 18, scope: !465)
!469 = !DILocation(line: 151, column: 14, scope: !465)
!470 = !DILocation(line: 151, column: 28, scope: !465)
!471 = !DILocation(line: 151, column: 35, scope: !465)
!472 = !DILocation(line: 151, column: 33, scope: !465)
!473 = !DILocation(line: 151, column: 9, scope: !465)
!474 = !DILocation(line: 152, column: 17, scope: !465)
!475 = !DILocation(line: 152, column: 34, scope: !465)
!476 = !DILocation(line: 152, column: 31, scope: !465)
!477 = !DILocation(line: 152, column: 23, scope: !465)
!478 = !DILocation(line: 154, column: 21, scope: !465)
!479 = !DILocation(line: 154, column: 55, scope: !465)
!480 = !DILocation(line: 154, column: 53, scope: !465)
!481 = !DILocation(line: 154, column: 97, scope: !465)
!482 = !DILocation(line: 154, column: 37, scope: !465)
!483 = !DILocation(line: 156, column: 21, scope: !465)
!484 = !DILocation(line: 158, column: 78, scope: !465)
!485 = !DILocation(line: 158, column: 35, scope: !465)
!486 = !DILocation(line: 158, column: 33, scope: !465)
!487 = !DILocation(line: 159, column: 13, scope: !465)
!488 = !DILocation(line: 161, column: 21, scope: !465)
!489 = !DILocation(line: 161, column: 55, scope: !465)
!490 = !DILocation(line: 161, column: 53, scope: !465)
!491 = !DILocation(line: 161, column: 98, scope: !465)
!492 = !DILocation(line: 161, column: 37, scope: !465)
!493 = !DILocation(line: 163, column: 21, scope: !465)
!494 = !DILocation(line: 165, column: 79, scope: !465)
!495 = !DILocation(line: 165, column: 35, scope: !465)
!496 = !DILocation(line: 165, column: 33, scope: !465)
!497 = !DILocation(line: 167, column: 36, scope: !465)
!498 = !DILocation(line: 167, column: 44, scope: !465)
!499 = !DILocation(line: 167, column: 51, scope: !465)
!500 = !DILocation(line: 167, column: 41, scope: !465)
!501 = !DILocation(line: 167, column: 17, scope: !465)
!502 = !DILocation(line: 167, column: 34, scope: !465)
!503 = !DILocation(line: 168, column: 13, scope: !465)
!504 = !DILocation(line: 170, column: 13, scope: !465)
!505 = !DILocation(line: 171, column: 9, scope: !465)
!506 = !DILocation(line: 151, column: 43, scope: !465)
!507 = distinct !{!507, !473, !505}
!508 = !DILocation(line: 172, column: 13, scope: !465)
!509 = !DILocation(line: 172, column: 30, scope: !465)
!510 = !DILocation(line: 173, column: 16, scope: !465)
!511 = !DILocation(line: 173, column: 9, scope: !465)
!512 = distinct !DISubprogram(name: "requiredSpace4VecElem<bool, ON>", scope: !1, file: !1, line: 113, type: !16, scopeLine: 113, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!513 = !DILocation(line: 114, column: 23, scope: !512)
!514 = !DILocation(line: 114, column: 16, scope: !512)
!515 = !DILocation(line: 114, column: 58, scope: !512)
!516 = !DILocation(line: 114, column: 51, scope: !512)
!517 = !DILocation(line: 114, column: 12, scope: !512)
!518 = !DILocation(line: 114, column: 5, scope: !512)
!519 = distinct !DISubprogram(name: "writeFormat4VecElem<bool, ON>", scope: !1, file: !1, line: 223, type: !16, scopeLine: 223, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!520 = !DILocation(line: 224, column: 60, scope: !519)
!521 = !DILocation(line: 224, column: 41, scope: !519)
!522 = !DILocation(line: 224, column: 86, scope: !519)
!523 = !DILocation(line: 225, column: 56, scope: !519)
!524 = !DILocation(line: 225, column: 54, scope: !519)
!525 = !DILocation(line: 225, column: 72, scope: !519)
!526 = !DILocation(line: 224, column: 28, scope: !519)
!527 = !DILocation(line: 224, column: 25, scope: !519)
!528 = !DILocation(line: 226, column: 16, scope: !519)
!529 = !DILocation(line: 226, column: 9, scope: !519)
!530 = distinct !DISubprogram(name: "requiredSpace4VecElem<bool, OFF>", scope: !1, file: !1, line: 117, type: !16, scopeLine: 117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!531 = !DILocation(line: 117, column: 79, scope: !530)
!532 = !DILocation(line: 117, column: 72, scope: !530)
!533 = distinct !DISubprogram(name: "writeFormat4VecElem<bool, OFF>", scope: !1, file: !1, line: 231, type: !16, scopeLine: 231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!534 = !DILocation(line: 232, column: 62, scope: !533)
!535 = !DILocation(line: 232, column: 94, scope: !533)
!536 = !DILocation(line: 232, column: 92, scope: !533)
!537 = !DILocation(line: 232, column: 110, scope: !533)
!538 = !DILocation(line: 232, column: 28, scope: !533)
!539 = !DILocation(line: 232, column: 25, scope: !533)
!540 = !DILocation(line: 233, column: 16, scope: !533)
!541 = !DILocation(line: 233, column: 9, scope: !533)
!542 = distinct !DISubprogram(name: "WriteArg<bool>", scope: !1, file: !1, line: 205, type: !16, scopeLine: 205, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!543 = !DILocation(line: 205, column: 47, scope: !542)
!544 = distinct !DISubprogram(name: "max<int>", scope: !1, file: !1, line: 84, type: !16, scopeLine: 84, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!545 = !DILocation(line: 85, column: 9, scope: !544)
!546 = !DILocation(line: 85, column: 13, scope: !544)
!547 = !DILocation(line: 85, column: 11, scope: !544)
!548 = !DILocation(line: 86, column: 16, scope: !544)
!549 = !DILocation(line: 86, column: 9, scope: !544)
!550 = !DILocation(line: 87, column: 12, scope: !544)
!551 = !DILocation(line: 87, column: 5, scope: !544)
!552 = !DILocation(line: 88, column: 1, scope: !544)
!553 = distinct !DISubprogram(name: "strLen<const char *>", scope: !1, file: !1, line: 76, type: !16, scopeLine: 76, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!554 = !DILocation(line: 77, column: 9, scope: !553)
!555 = !DILocation(line: 78, column: 5, scope: !553)
!556 = !DILocation(line: 78, column: 12, scope: !553)
!557 = !DILocation(line: 78, column: 16, scope: !553)
!558 = !DILocation(line: 78, column: 21, scope: !553)
!559 = !DILocation(line: 78, column: 30, scope: !553)
!560 = distinct !{!560, !555, !561}
!561 = !DILocation(line: 79, column: 9, scope: !553)
!562 = !DILocation(line: 80, column: 12, scope: !553)
!563 = !DILocation(line: 80, column: 5, scope: !553)
!564 = distinct !DISubprogram(name: "UniArg2StrIfSuitable<int, ArgWriter>", scope: !74, file: !74, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!565 = !DILocation(line: 82, column: 9, scope: !564)
!566 = !DILocation(line: 82, column: 17, scope: !564)
!567 = !DILocation(line: 82, column: 14, scope: !564)
!568 = !DILocation(line: 83, column: 15, scope: !564)
!569 = !DILocation(line: 83, column: 34, scope: !564)
!570 = !DILocation(line: 83, column: 13, scope: !564)
!571 = !DILocation(line: 84, column: 9, scope: !564)
!572 = !DILocation(line: 86, column: 5, scope: !564)
!573 = !DILocation(line: 87, column: 1, scope: !564)
!574 = distinct !DISubprogram(name: "VarArg2StrIfSuitable<int, ArgWriter>", scope: !74, file: !74, line: 92, type: !16, scopeLine: 92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!575 = !DILocation(line: 93, column: 9, scope: !574)
!576 = !DILocation(line: 93, column: 17, scope: !574)
!577 = !DILocation(line: 93, column: 14, scope: !574)
!578 = !DILocation(line: 94, column: 15, scope: !574)
!579 = !DILocation(line: 94, column: 34, scope: !574)
!580 = !DILocation(line: 94, column: 13, scope: !574)
!581 = !DILocation(line: 95, column: 9, scope: !574)
!582 = !DILocation(line: 97, column: 5, scope: !574)
!583 = !DILocation(line: 98, column: 1, scope: !574)
!584 = distinct !DISubprogram(name: "getEncoding4Uniform<int>", scope: !192, file: !192, line: 76, type: !16, scopeLine: 76, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!585 = !DILocation(line: 76, column: 68, scope: !584)
!586 = distinct !DISubprogram(name: "uniform2Str<int>", scope: !1, file: !1, line: 130, type: !16, scopeLine: 130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!587 = !DILocation(line: 131, column: 20, scope: !586)
!588 = !DILocation(line: 131, column: 14, scope: !586)
!589 = !DILocation(line: 133, column: 40, scope: !586)
!590 = !DILocation(line: 133, column: 27, scope: !586)
!591 = !DILocation(line: 133, column: 14, scope: !586)
!592 = !DILocation(line: 134, column: 13, scope: !586)
!593 = !DILocation(line: 134, column: 25, scope: !586)
!594 = !DILocation(line: 136, column: 9, scope: !586)
!595 = !DILocation(line: 137, column: 16, scope: !586)
!596 = !DILocation(line: 137, column: 9, scope: !586)
!597 = distinct !DISubprogram(name: "cmType2Specifier<int>", scope: !1, file: !1, line: 90, type: !16, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!598 = !DILocation(line: 90, column: 70, scope: !597)
!599 = !DILocation(line: 90, column: 63, scope: !597)
!600 = distinct !DISubprogram(name: "WriteArg<int>", scope: !1, file: !1, line: 181, type: !16, scopeLine: 181, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!601 = !DILocation(line: 187, column: 19, scope: !600)
!602 = !DILocation(line: 187, column: 17, scope: !600)
!603 = !DILocation(line: 188, column: 18, scope: !600)
!604 = !DILocation(line: 190, column: 28, scope: !600)
!605 = !DILocation(line: 190, column: 22, scope: !600)
!606 = !DILocation(line: 191, column: 40, scope: !600)
!607 = !DILocation(line: 191, column: 45, scope: !600)
!608 = !DILocation(line: 191, column: 54, scope: !600)
!609 = !DILocation(line: 191, column: 59, scope: !600)
!610 = !DILocation(line: 191, column: 9, scope: !600)
!611 = !DILocation(line: 192, column: 9, scope: !600)
!612 = !DILocation(line: 192, column: 17, scope: !600)
!613 = !DILocation(line: 193, column: 5, scope: !600)
!614 = distinct !DISubprogram(name: "type2Specifier<int>", scope: !192, file: !192, line: 119, type: !16, scopeLine: 119, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!615 = !DILocation(line: 119, column: 56, scope: !614)
!616 = !{i32 6922}
!617 = distinct !DISubprogram(name: "getEncoding4Varying<int>", scope: !192, file: !192, line: 85, type: !16, scopeLine: 85, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!618 = !DILocation(line: 85, column: 68, scope: !617)
!619 = distinct !DISubprogram(name: "varying2Str<int>", scope: !1, file: !1, line: 147, type: !16, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!620 = !DILocation(line: 149, column: 16, scope: !619)
!621 = !DILocation(line: 150, column: 13, scope: !619)
!622 = !DILocation(line: 151, column: 18, scope: !619)
!623 = !DILocation(line: 151, column: 14, scope: !619)
!624 = !DILocation(line: 151, column: 28, scope: !619)
!625 = !DILocation(line: 151, column: 35, scope: !619)
!626 = !DILocation(line: 151, column: 33, scope: !619)
!627 = !DILocation(line: 151, column: 9, scope: !619)
!628 = !DILocation(line: 152, column: 17, scope: !619)
!629 = !DILocation(line: 152, column: 34, scope: !619)
!630 = !DILocation(line: 152, column: 31, scope: !619)
!631 = !DILocation(line: 152, column: 23, scope: !619)
!632 = !DILocation(line: 154, column: 21, scope: !619)
!633 = !DILocation(line: 154, column: 55, scope: !619)
!634 = !DILocation(line: 154, column: 53, scope: !619)
!635 = !DILocation(line: 154, column: 97, scope: !619)
!636 = !DILocation(line: 154, column: 37, scope: !619)
!637 = !DILocation(line: 156, column: 21, scope: !619)
!638 = !DILocation(line: 158, column: 78, scope: !619)
!639 = !DILocation(line: 158, column: 35, scope: !619)
!640 = !DILocation(line: 158, column: 33, scope: !619)
!641 = !DILocation(line: 159, column: 13, scope: !619)
!642 = !DILocation(line: 161, column: 21, scope: !619)
!643 = !DILocation(line: 161, column: 55, scope: !619)
!644 = !DILocation(line: 161, column: 53, scope: !619)
!645 = !DILocation(line: 161, column: 98, scope: !619)
!646 = !DILocation(line: 161, column: 37, scope: !619)
!647 = !DILocation(line: 163, column: 21, scope: !619)
!648 = !DILocation(line: 165, column: 79, scope: !619)
!649 = !DILocation(line: 165, column: 35, scope: !619)
!650 = !DILocation(line: 165, column: 33, scope: !619)
!651 = !DILocation(line: 167, column: 36, scope: !619)
!652 = !DILocation(line: 167, column: 44, scope: !619)
!653 = !DILocation(line: 167, column: 51, scope: !619)
!654 = !DILocation(line: 167, column: 41, scope: !619)
!655 = !DILocation(line: 167, column: 17, scope: !619)
!656 = !DILocation(line: 167, column: 34, scope: !619)
!657 = !DILocation(line: 168, column: 13, scope: !619)
!658 = !DILocation(line: 170, column: 13, scope: !619)
!659 = !DILocation(line: 171, column: 9, scope: !619)
!660 = !DILocation(line: 151, column: 43, scope: !619)
!661 = distinct !{!661, !627, !659}
!662 = !DILocation(line: 172, column: 13, scope: !619)
!663 = !DILocation(line: 172, column: 30, scope: !619)
!664 = !DILocation(line: 173, column: 16, scope: !619)
!665 = !DILocation(line: 173, column: 9, scope: !619)
!666 = distinct !DISubprogram(name: "requiredSpace4VecElem<int, ON>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!667 = !DILocation(line: 103, column: 32, scope: !666)
!668 = !DILocation(line: 103, column: 25, scope: !666)
!669 = !DILocation(line: 103, column: 10, scope: !666)
!670 = !DILocation(line: 106, column: 16, scope: !666)
!671 = !DILocation(line: 106, column: 29, scope: !666)
!672 = !DILocation(line: 106, column: 9, scope: !666)
!673 = distinct !DISubprogram(name: "writeFormat4VecElem<int, ON>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!674 = !DILocation(line: 212, column: 20, scope: !673)
!675 = !DILocation(line: 212, column: 14, scope: !673)
!676 = !DILocation(line: 215, column: 41, scope: !673)
!677 = !DILocation(line: 215, column: 51, scope: !673)
!678 = !DILocation(line: 215, column: 83, scope: !673)
!679 = !DILocation(line: 215, column: 81, scope: !673)
!680 = !DILocation(line: 215, column: 99, scope: !673)
!681 = !DILocation(line: 215, column: 28, scope: !673)
!682 = !DILocation(line: 215, column: 25, scope: !673)
!683 = !DILocation(line: 218, column: 16, scope: !673)
!684 = !DILocation(line: 218, column: 9, scope: !673)
!685 = distinct !DISubprogram(name: "requiredSpace4VecElem<int, OFF>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!686 = !DILocation(line: 103, column: 32, scope: !685)
!687 = !DILocation(line: 103, column: 25, scope: !685)
!688 = !DILocation(line: 103, column: 10, scope: !685)
!689 = !DILocation(line: 109, column: 16, scope: !685)
!690 = !DILocation(line: 109, column: 29, scope: !685)
!691 = !DILocation(line: 109, column: 9, scope: !685)
!692 = distinct !DISubprogram(name: "writeFormat4VecElem<int, OFF>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!693 = !DILocation(line: 212, column: 20, scope: !692)
!694 = !DILocation(line: 212, column: 14, scope: !692)
!695 = !DILocation(line: 214, column: 56, scope: !692)
!696 = !DILocation(line: 214, column: 88, scope: !692)
!697 = !DILocation(line: 214, column: 86, scope: !692)
!698 = !DILocation(line: 214, column: 104, scope: !692)
!699 = !DILocation(line: 214, column: 32, scope: !692)
!700 = !DILocation(line: 214, column: 29, scope: !692)
!701 = !DILocation(line: 215, column: 41, scope: !692)
!702 = !DILocation(line: 215, column: 51, scope: !692)
!703 = !DILocation(line: 215, column: 83, scope: !692)
!704 = !DILocation(line: 215, column: 81, scope: !692)
!705 = !DILocation(line: 215, column: 99, scope: !692)
!706 = !DILocation(line: 215, column: 28, scope: !692)
!707 = !DILocation(line: 215, column: 25, scope: !692)
!708 = !DILocation(line: 217, column: 56, scope: !692)
!709 = !DILocation(line: 217, column: 88, scope: !692)
!710 = !DILocation(line: 217, column: 86, scope: !692)
!711 = !DILocation(line: 217, column: 104, scope: !692)
!712 = !DILocation(line: 217, column: 32, scope: !692)
!713 = !DILocation(line: 217, column: 29, scope: !692)
!714 = !DILocation(line: 218, column: 16, scope: !692)
!715 = !DILocation(line: 218, column: 9, scope: !692)
!716 = distinct !DISubprogram(name: "UniArg2StrIfSuitable<unsigned int, ArgWriter>", scope: !74, file: !74, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!717 = !DILocation(line: 82, column: 9, scope: !716)
!718 = !DILocation(line: 82, column: 17, scope: !716)
!719 = !DILocation(line: 82, column: 14, scope: !716)
!720 = !DILocation(line: 83, column: 15, scope: !716)
!721 = !DILocation(line: 83, column: 34, scope: !716)
!722 = !DILocation(line: 83, column: 13, scope: !716)
!723 = !DILocation(line: 84, column: 9, scope: !716)
!724 = !DILocation(line: 86, column: 5, scope: !716)
!725 = !DILocation(line: 87, column: 1, scope: !716)
!726 = distinct !DISubprogram(name: "VarArg2StrIfSuitable<unsigned int, ArgWriter>", scope: !74, file: !74, line: 92, type: !16, scopeLine: 92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!727 = !DILocation(line: 93, column: 9, scope: !726)
!728 = !DILocation(line: 93, column: 17, scope: !726)
!729 = !DILocation(line: 93, column: 14, scope: !726)
!730 = !DILocation(line: 94, column: 15, scope: !726)
!731 = !DILocation(line: 94, column: 34, scope: !726)
!732 = !DILocation(line: 94, column: 13, scope: !726)
!733 = !DILocation(line: 95, column: 9, scope: !726)
!734 = !DILocation(line: 97, column: 5, scope: !726)
!735 = !DILocation(line: 98, column: 1, scope: !726)
!736 = distinct !DISubprogram(name: "getEncoding4Uniform<unsigned int>", scope: !192, file: !192, line: 77, type: !16, scopeLine: 77, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!737 = !DILocation(line: 77, column: 73, scope: !736)
!738 = distinct !DISubprogram(name: "uniform2Str<unsigned int>", scope: !1, file: !1, line: 130, type: !16, scopeLine: 130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!739 = !DILocation(line: 131, column: 20, scope: !738)
!740 = !DILocation(line: 131, column: 14, scope: !738)
!741 = !DILocation(line: 133, column: 40, scope: !738)
!742 = !DILocation(line: 133, column: 27, scope: !738)
!743 = !DILocation(line: 133, column: 14, scope: !738)
!744 = !DILocation(line: 134, column: 13, scope: !738)
!745 = !DILocation(line: 134, column: 25, scope: !738)
!746 = !DILocation(line: 136, column: 9, scope: !738)
!747 = !DILocation(line: 137, column: 16, scope: !738)
!748 = !DILocation(line: 137, column: 9, scope: !738)
!749 = distinct !DISubprogram(name: "cmType2Specifier<unsigned int>", scope: !1, file: !1, line: 90, type: !16, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!750 = !DILocation(line: 90, column: 70, scope: !749)
!751 = !DILocation(line: 90, column: 63, scope: !749)
!752 = distinct !DISubprogram(name: "WriteArg<unsigned int>", scope: !1, file: !1, line: 181, type: !16, scopeLine: 181, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!753 = !DILocation(line: 187, column: 19, scope: !752)
!754 = !DILocation(line: 187, column: 17, scope: !752)
!755 = !DILocation(line: 188, column: 18, scope: !752)
!756 = !DILocation(line: 190, column: 28, scope: !752)
!757 = !DILocation(line: 190, column: 22, scope: !752)
!758 = !DILocation(line: 191, column: 40, scope: !752)
!759 = !DILocation(line: 191, column: 45, scope: !752)
!760 = !DILocation(line: 191, column: 54, scope: !752)
!761 = !DILocation(line: 191, column: 59, scope: !752)
!762 = !DILocation(line: 191, column: 9, scope: !752)
!763 = !DILocation(line: 192, column: 9, scope: !752)
!764 = !DILocation(line: 192, column: 17, scope: !752)
!765 = !DILocation(line: 193, column: 5, scope: !752)
!766 = distinct !DISubprogram(name: "type2Specifier<unsigned int>", scope: !192, file: !192, line: 120, type: !16, scopeLine: 120, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!767 = !DILocation(line: 120, column: 61, scope: !766)
!768 = distinct !DISubprogram(name: "getEncoding4Varying<unsigned int>", scope: !192, file: !192, line: 86, type: !16, scopeLine: 86, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!769 = !DILocation(line: 86, column: 73, scope: !768)
!770 = distinct !DISubprogram(name: "varying2Str<unsigned int>", scope: !1, file: !1, line: 147, type: !16, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!771 = !DILocation(line: 149, column: 16, scope: !770)
!772 = !DILocation(line: 150, column: 13, scope: !770)
!773 = !DILocation(line: 151, column: 18, scope: !770)
!774 = !DILocation(line: 151, column: 14, scope: !770)
!775 = !DILocation(line: 151, column: 28, scope: !770)
!776 = !DILocation(line: 151, column: 35, scope: !770)
!777 = !DILocation(line: 151, column: 33, scope: !770)
!778 = !DILocation(line: 151, column: 9, scope: !770)
!779 = !DILocation(line: 152, column: 17, scope: !770)
!780 = !DILocation(line: 152, column: 34, scope: !770)
!781 = !DILocation(line: 152, column: 31, scope: !770)
!782 = !DILocation(line: 152, column: 23, scope: !770)
!783 = !DILocation(line: 154, column: 21, scope: !770)
!784 = !DILocation(line: 154, column: 55, scope: !770)
!785 = !DILocation(line: 154, column: 53, scope: !770)
!786 = !DILocation(line: 154, column: 97, scope: !770)
!787 = !DILocation(line: 154, column: 37, scope: !770)
!788 = !DILocation(line: 156, column: 21, scope: !770)
!789 = !DILocation(line: 158, column: 78, scope: !770)
!790 = !DILocation(line: 158, column: 35, scope: !770)
!791 = !DILocation(line: 158, column: 33, scope: !770)
!792 = !DILocation(line: 159, column: 13, scope: !770)
!793 = !DILocation(line: 161, column: 21, scope: !770)
!794 = !DILocation(line: 161, column: 55, scope: !770)
!795 = !DILocation(line: 161, column: 53, scope: !770)
!796 = !DILocation(line: 161, column: 98, scope: !770)
!797 = !DILocation(line: 161, column: 37, scope: !770)
!798 = !DILocation(line: 163, column: 21, scope: !770)
!799 = !DILocation(line: 165, column: 79, scope: !770)
!800 = !DILocation(line: 165, column: 35, scope: !770)
!801 = !DILocation(line: 165, column: 33, scope: !770)
!802 = !DILocation(line: 167, column: 36, scope: !770)
!803 = !DILocation(line: 167, column: 44, scope: !770)
!804 = !DILocation(line: 167, column: 51, scope: !770)
!805 = !DILocation(line: 167, column: 41, scope: !770)
!806 = !DILocation(line: 167, column: 17, scope: !770)
!807 = !DILocation(line: 167, column: 34, scope: !770)
!808 = !DILocation(line: 168, column: 13, scope: !770)
!809 = !DILocation(line: 170, column: 13, scope: !770)
!810 = !DILocation(line: 171, column: 9, scope: !770)
!811 = !DILocation(line: 151, column: 43, scope: !770)
!812 = distinct !{!812, !778, !810}
!813 = !DILocation(line: 172, column: 13, scope: !770)
!814 = !DILocation(line: 172, column: 30, scope: !770)
!815 = !DILocation(line: 173, column: 16, scope: !770)
!816 = !DILocation(line: 173, column: 9, scope: !770)
!817 = distinct !DISubprogram(name: "requiredSpace4VecElem<unsigned int, ON>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!818 = !DILocation(line: 103, column: 32, scope: !817)
!819 = !DILocation(line: 103, column: 25, scope: !817)
!820 = !DILocation(line: 103, column: 10, scope: !817)
!821 = !DILocation(line: 106, column: 16, scope: !817)
!822 = !DILocation(line: 106, column: 29, scope: !817)
!823 = !DILocation(line: 106, column: 9, scope: !817)
!824 = distinct !DISubprogram(name: "writeFormat4VecElem<unsigned int, ON>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!825 = !DILocation(line: 212, column: 20, scope: !824)
!826 = !DILocation(line: 212, column: 14, scope: !824)
!827 = !DILocation(line: 215, column: 41, scope: !824)
!828 = !DILocation(line: 215, column: 51, scope: !824)
!829 = !DILocation(line: 215, column: 83, scope: !824)
!830 = !DILocation(line: 215, column: 81, scope: !824)
!831 = !DILocation(line: 215, column: 99, scope: !824)
!832 = !DILocation(line: 215, column: 28, scope: !824)
!833 = !DILocation(line: 215, column: 25, scope: !824)
!834 = !DILocation(line: 218, column: 16, scope: !824)
!835 = !DILocation(line: 218, column: 9, scope: !824)
!836 = distinct !DISubprogram(name: "requiredSpace4VecElem<unsigned int, OFF>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!837 = !DILocation(line: 103, column: 32, scope: !836)
!838 = !DILocation(line: 103, column: 25, scope: !836)
!839 = !DILocation(line: 103, column: 10, scope: !836)
!840 = !DILocation(line: 109, column: 16, scope: !836)
!841 = !DILocation(line: 109, column: 29, scope: !836)
!842 = !DILocation(line: 109, column: 9, scope: !836)
!843 = distinct !DISubprogram(name: "writeFormat4VecElem<unsigned int, OFF>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!844 = !DILocation(line: 212, column: 20, scope: !843)
!845 = !DILocation(line: 212, column: 14, scope: !843)
!846 = !DILocation(line: 214, column: 56, scope: !843)
!847 = !DILocation(line: 214, column: 88, scope: !843)
!848 = !DILocation(line: 214, column: 86, scope: !843)
!849 = !DILocation(line: 214, column: 104, scope: !843)
!850 = !DILocation(line: 214, column: 32, scope: !843)
!851 = !DILocation(line: 214, column: 29, scope: !843)
!852 = !DILocation(line: 215, column: 41, scope: !843)
!853 = !DILocation(line: 215, column: 51, scope: !843)
!854 = !DILocation(line: 215, column: 83, scope: !843)
!855 = !DILocation(line: 215, column: 81, scope: !843)
!856 = !DILocation(line: 215, column: 99, scope: !843)
!857 = !DILocation(line: 215, column: 28, scope: !843)
!858 = !DILocation(line: 215, column: 25, scope: !843)
!859 = !DILocation(line: 217, column: 56, scope: !843)
!860 = !DILocation(line: 217, column: 88, scope: !843)
!861 = !DILocation(line: 217, column: 86, scope: !843)
!862 = !DILocation(line: 217, column: 104, scope: !843)
!863 = !DILocation(line: 217, column: 32, scope: !843)
!864 = !DILocation(line: 217, column: 29, scope: !843)
!865 = !DILocation(line: 218, column: 16, scope: !843)
!866 = !DILocation(line: 218, column: 9, scope: !843)
!867 = distinct !DISubprogram(name: "UniArg2StrIfSuitable<float, ArgWriter>", scope: !74, file: !74, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!868 = !DILocation(line: 82, column: 9, scope: !867)
!869 = !DILocation(line: 82, column: 17, scope: !867)
!870 = !DILocation(line: 82, column: 14, scope: !867)
!871 = !DILocation(line: 83, column: 15, scope: !867)
!872 = !DILocation(line: 83, column: 34, scope: !867)
!873 = !DILocation(line: 83, column: 13, scope: !867)
!874 = !DILocation(line: 84, column: 9, scope: !867)
!875 = !DILocation(line: 86, column: 5, scope: !867)
!876 = !DILocation(line: 87, column: 1, scope: !867)
!877 = distinct !DISubprogram(name: "VarArg2StrIfSuitable<float, ArgWriter>", scope: !74, file: !74, line: 92, type: !16, scopeLine: 92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!878 = !DILocation(line: 93, column: 9, scope: !877)
!879 = !DILocation(line: 93, column: 17, scope: !877)
!880 = !DILocation(line: 93, column: 14, scope: !877)
!881 = !DILocation(line: 94, column: 15, scope: !877)
!882 = !DILocation(line: 94, column: 34, scope: !877)
!883 = !DILocation(line: 94, column: 13, scope: !877)
!884 = !DILocation(line: 95, column: 9, scope: !877)
!885 = !DILocation(line: 97, column: 5, scope: !877)
!886 = !DILocation(line: 98, column: 1, scope: !877)
!887 = distinct !DISubprogram(name: "getEncoding4Uniform<float>", scope: !192, file: !192, line: 78, type: !16, scopeLine: 78, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!888 = !DILocation(line: 78, column: 70, scope: !887)
!889 = distinct !DISubprogram(name: "uniform2Str<float>", scope: !1, file: !1, line: 130, type: !16, scopeLine: 130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!890 = !DILocation(line: 131, column: 20, scope: !889)
!891 = !DILocation(line: 131, column: 14, scope: !889)
!892 = !DILocation(line: 133, column: 40, scope: !889)
!893 = !DILocation(line: 133, column: 27, scope: !889)
!894 = !DILocation(line: 133, column: 14, scope: !889)
!895 = !DILocation(line: 134, column: 13, scope: !889)
!896 = !DILocation(line: 134, column: 25, scope: !889)
!897 = !DILocation(line: 136, column: 9, scope: !889)
!898 = !DILocation(line: 137, column: 16, scope: !889)
!899 = !DILocation(line: 137, column: 9, scope: !889)
!900 = distinct !DISubprogram(name: "cmType2Specifier<float>", scope: !1, file: !1, line: 90, type: !16, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!901 = !DILocation(line: 90, column: 70, scope: !900)
!902 = !DILocation(line: 90, column: 63, scope: !900)
!903 = distinct !DISubprogram(name: "WriteArg<float>", scope: !1, file: !1, line: 181, type: !16, scopeLine: 181, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!904 = !DILocation(line: 187, column: 19, scope: !903)
!905 = !DILocation(line: 187, column: 17, scope: !903)
!906 = !DILocation(line: 188, column: 18, scope: !903)
!907 = !DILocation(line: 190, column: 28, scope: !903)
!908 = !DILocation(line: 190, column: 22, scope: !903)
!909 = !DILocation(line: 191, column: 40, scope: !903)
!910 = !DILocation(line: 191, column: 45, scope: !903)
!911 = !DILocation(line: 191, column: 54, scope: !903)
!912 = !DILocation(line: 191, column: 59, scope: !903)
!913 = !DILocation(line: 191, column: 9, scope: !903)
!914 = !DILocation(line: 192, column: 9, scope: !903)
!915 = !DILocation(line: 192, column: 17, scope: !903)
!916 = !DILocation(line: 193, column: 5, scope: !903)
!917 = distinct !DISubprogram(name: "type2Specifier<float>", scope: !192, file: !192, line: 121, type: !16, scopeLine: 121, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!918 = !DILocation(line: 121, column: 58, scope: !917)
!919 = distinct !DISubprogram(name: "getEncoding4Varying<float>", scope: !192, file: !192, line: 87, type: !16, scopeLine: 87, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!920 = !DILocation(line: 87, column: 70, scope: !919)
!921 = distinct !DISubprogram(name: "varying2Str<float>", scope: !1, file: !1, line: 147, type: !16, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!922 = !DILocation(line: 149, column: 16, scope: !921)
!923 = !DILocation(line: 150, column: 13, scope: !921)
!924 = !DILocation(line: 151, column: 18, scope: !921)
!925 = !DILocation(line: 151, column: 14, scope: !921)
!926 = !DILocation(line: 151, column: 28, scope: !921)
!927 = !DILocation(line: 151, column: 35, scope: !921)
!928 = !DILocation(line: 151, column: 33, scope: !921)
!929 = !DILocation(line: 151, column: 9, scope: !921)
!930 = !DILocation(line: 152, column: 17, scope: !921)
!931 = !DILocation(line: 152, column: 34, scope: !921)
!932 = !DILocation(line: 152, column: 31, scope: !921)
!933 = !DILocation(line: 152, column: 23, scope: !921)
!934 = !DILocation(line: 154, column: 21, scope: !921)
!935 = !DILocation(line: 154, column: 55, scope: !921)
!936 = !DILocation(line: 154, column: 53, scope: !921)
!937 = !DILocation(line: 154, column: 97, scope: !921)
!938 = !DILocation(line: 154, column: 37, scope: !921)
!939 = !DILocation(line: 156, column: 21, scope: !921)
!940 = !DILocation(line: 158, column: 78, scope: !921)
!941 = !DILocation(line: 158, column: 35, scope: !921)
!942 = !DILocation(line: 158, column: 33, scope: !921)
!943 = !DILocation(line: 159, column: 13, scope: !921)
!944 = !DILocation(line: 161, column: 21, scope: !921)
!945 = !DILocation(line: 161, column: 55, scope: !921)
!946 = !DILocation(line: 161, column: 53, scope: !921)
!947 = !DILocation(line: 161, column: 98, scope: !921)
!948 = !DILocation(line: 161, column: 37, scope: !921)
!949 = !DILocation(line: 163, column: 21, scope: !921)
!950 = !DILocation(line: 165, column: 79, scope: !921)
!951 = !DILocation(line: 165, column: 35, scope: !921)
!952 = !DILocation(line: 165, column: 33, scope: !921)
!953 = !DILocation(line: 167, column: 36, scope: !921)
!954 = !DILocation(line: 167, column: 44, scope: !921)
!955 = !DILocation(line: 167, column: 51, scope: !921)
!956 = !DILocation(line: 167, column: 41, scope: !921)
!957 = !DILocation(line: 167, column: 17, scope: !921)
!958 = !DILocation(line: 167, column: 34, scope: !921)
!959 = !DILocation(line: 168, column: 13, scope: !921)
!960 = !DILocation(line: 170, column: 13, scope: !921)
!961 = !DILocation(line: 171, column: 9, scope: !921)
!962 = !DILocation(line: 151, column: 43, scope: !921)
!963 = distinct !{!963, !929, !961}
!964 = !DILocation(line: 172, column: 13, scope: !921)
!965 = !DILocation(line: 172, column: 30, scope: !921)
!966 = !DILocation(line: 173, column: 16, scope: !921)
!967 = !DILocation(line: 173, column: 9, scope: !921)
!968 = distinct !DISubprogram(name: "requiredSpace4VecElem<float, ON>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!969 = !DILocation(line: 103, column: 32, scope: !968)
!970 = !DILocation(line: 103, column: 25, scope: !968)
!971 = !DILocation(line: 103, column: 10, scope: !968)
!972 = !DILocation(line: 106, column: 16, scope: !968)
!973 = !DILocation(line: 106, column: 29, scope: !968)
!974 = !DILocation(line: 106, column: 9, scope: !968)
!975 = distinct !DISubprogram(name: "writeFormat4VecElem<float, ON>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!976 = !DILocation(line: 212, column: 20, scope: !975)
!977 = !DILocation(line: 212, column: 14, scope: !975)
!978 = !DILocation(line: 215, column: 41, scope: !975)
!979 = !DILocation(line: 215, column: 51, scope: !975)
!980 = !DILocation(line: 215, column: 83, scope: !975)
!981 = !DILocation(line: 215, column: 81, scope: !975)
!982 = !DILocation(line: 215, column: 99, scope: !975)
!983 = !DILocation(line: 215, column: 28, scope: !975)
!984 = !DILocation(line: 215, column: 25, scope: !975)
!985 = !DILocation(line: 218, column: 16, scope: !975)
!986 = !DILocation(line: 218, column: 9, scope: !975)
!987 = distinct !DISubprogram(name: "requiredSpace4VecElem<float, OFF>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!988 = !DILocation(line: 103, column: 32, scope: !987)
!989 = !DILocation(line: 103, column: 25, scope: !987)
!990 = !DILocation(line: 103, column: 10, scope: !987)
!991 = !DILocation(line: 109, column: 16, scope: !987)
!992 = !DILocation(line: 109, column: 29, scope: !987)
!993 = !DILocation(line: 109, column: 9, scope: !987)
!994 = distinct !DISubprogram(name: "writeFormat4VecElem<float, OFF>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!995 = !DILocation(line: 212, column: 20, scope: !994)
!996 = !DILocation(line: 212, column: 14, scope: !994)
!997 = !DILocation(line: 214, column: 56, scope: !994)
!998 = !DILocation(line: 214, column: 88, scope: !994)
!999 = !DILocation(line: 214, column: 86, scope: !994)
!1000 = !DILocation(line: 214, column: 104, scope: !994)
!1001 = !DILocation(line: 214, column: 32, scope: !994)
!1002 = !DILocation(line: 214, column: 29, scope: !994)
!1003 = !DILocation(line: 215, column: 41, scope: !994)
!1004 = !DILocation(line: 215, column: 51, scope: !994)
!1005 = !DILocation(line: 215, column: 83, scope: !994)
!1006 = !DILocation(line: 215, column: 81, scope: !994)
!1007 = !DILocation(line: 215, column: 99, scope: !994)
!1008 = !DILocation(line: 215, column: 28, scope: !994)
!1009 = !DILocation(line: 215, column: 25, scope: !994)
!1010 = !DILocation(line: 217, column: 56, scope: !994)
!1011 = !DILocation(line: 217, column: 88, scope: !994)
!1012 = !DILocation(line: 217, column: 86, scope: !994)
!1013 = !DILocation(line: 217, column: 104, scope: !994)
!1014 = !DILocation(line: 217, column: 32, scope: !994)
!1015 = !DILocation(line: 217, column: 29, scope: !994)
!1016 = !DILocation(line: 218, column: 16, scope: !994)
!1017 = !DILocation(line: 218, column: 9, scope: !994)
!1018 = distinct !DISubprogram(name: "UniArg2StrIfSuitable<long long, ArgWriter>", scope: !74, file: !74, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1019 = !DILocation(line: 82, column: 9, scope: !1018)
!1020 = !DILocation(line: 82, column: 17, scope: !1018)
!1021 = !DILocation(line: 82, column: 14, scope: !1018)
!1022 = !DILocation(line: 83, column: 15, scope: !1018)
!1023 = !DILocation(line: 83, column: 34, scope: !1018)
!1024 = !DILocation(line: 83, column: 13, scope: !1018)
!1025 = !DILocation(line: 84, column: 9, scope: !1018)
!1026 = !DILocation(line: 86, column: 5, scope: !1018)
!1027 = !DILocation(line: 87, column: 1, scope: !1018)
!1028 = distinct !DISubprogram(name: "VarArg2StrIfSuitable<long long, ArgWriter>", scope: !74, file: !74, line: 92, type: !16, scopeLine: 92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1029 = !DILocation(line: 93, column: 9, scope: !1028)
!1030 = !DILocation(line: 93, column: 17, scope: !1028)
!1031 = !DILocation(line: 93, column: 14, scope: !1028)
!1032 = !DILocation(line: 94, column: 15, scope: !1028)
!1033 = !DILocation(line: 94, column: 34, scope: !1028)
!1034 = !DILocation(line: 94, column: 13, scope: !1028)
!1035 = !DILocation(line: 95, column: 9, scope: !1028)
!1036 = !DILocation(line: 97, column: 5, scope: !1028)
!1037 = !DILocation(line: 98, column: 1, scope: !1028)
!1038 = distinct !DISubprogram(name: "getEncoding4Uniform<long long>", scope: !192, file: !192, line: 79, type: !16, scopeLine: 79, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1039 = !DILocation(line: 79, column: 74, scope: !1038)
!1040 = distinct !DISubprogram(name: "uniform2Str<long long>", scope: !1, file: !1, line: 130, type: !16, scopeLine: 130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1041 = !DILocation(line: 131, column: 20, scope: !1040)
!1042 = !DILocation(line: 131, column: 14, scope: !1040)
!1043 = !DILocation(line: 133, column: 40, scope: !1040)
!1044 = !DILocation(line: 133, column: 27, scope: !1040)
!1045 = !DILocation(line: 133, column: 14, scope: !1040)
!1046 = !DILocation(line: 134, column: 13, scope: !1040)
!1047 = !DILocation(line: 134, column: 25, scope: !1040)
!1048 = !DILocation(line: 136, column: 9, scope: !1040)
!1049 = !DILocation(line: 137, column: 16, scope: !1040)
!1050 = !DILocation(line: 137, column: 9, scope: !1040)
!1051 = distinct !DISubprogram(name: "cmType2Specifier<long long>", scope: !1, file: !1, line: 90, type: !16, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1052 = !DILocation(line: 90, column: 70, scope: !1051)
!1053 = !DILocation(line: 90, column: 63, scope: !1051)
!1054 = distinct !DISubprogram(name: "WriteArg<long long>", scope: !1, file: !1, line: 181, type: !16, scopeLine: 181, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1055 = !DILocation(line: 184, column: 19, scope: !1054)
!1056 = !DILocation(line: 184, column: 17, scope: !1054)
!1057 = !DILocation(line: 185, column: 20, scope: !1054)
!1058 = !DILocation(line: 185, column: 18, scope: !1054)
!1059 = !DILocation(line: 190, column: 28, scope: !1054)
!1060 = !DILocation(line: 190, column: 22, scope: !1054)
!1061 = !DILocation(line: 191, column: 40, scope: !1054)
!1062 = !DILocation(line: 191, column: 45, scope: !1054)
!1063 = !DILocation(line: 191, column: 54, scope: !1054)
!1064 = !DILocation(line: 191, column: 59, scope: !1054)
!1065 = !DILocation(line: 191, column: 9, scope: !1054)
!1066 = !DILocation(line: 192, column: 9, scope: !1054)
!1067 = !DILocation(line: 192, column: 17, scope: !1054)
!1068 = !DILocation(line: 193, column: 5, scope: !1054)
!1069 = distinct !DISubprogram(name: "type2Specifier<long long>", scope: !192, file: !192, line: 122, type: !16, scopeLine: 122, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1070 = !DILocation(line: 122, column: 62, scope: !1069)
!1071 = distinct !DISubprogram(name: "getEncoding4Varying<long long>", scope: !192, file: !192, line: 88, type: !16, scopeLine: 88, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1072 = !DILocation(line: 88, column: 74, scope: !1071)
!1073 = distinct !DISubprogram(name: "varying2Str<long long>", scope: !1, file: !1, line: 147, type: !16, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1074 = !DILocation(line: 149, column: 16, scope: !1073)
!1075 = !DILocation(line: 150, column: 13, scope: !1073)
!1076 = !DILocation(line: 151, column: 18, scope: !1073)
!1077 = !DILocation(line: 151, column: 14, scope: !1073)
!1078 = !DILocation(line: 151, column: 28, scope: !1073)
!1079 = !DILocation(line: 151, column: 35, scope: !1073)
!1080 = !DILocation(line: 151, column: 33, scope: !1073)
!1081 = !DILocation(line: 151, column: 9, scope: !1073)
!1082 = !DILocation(line: 152, column: 17, scope: !1073)
!1083 = !DILocation(line: 152, column: 34, scope: !1073)
!1084 = !DILocation(line: 152, column: 31, scope: !1073)
!1085 = !DILocation(line: 152, column: 23, scope: !1073)
!1086 = !DILocation(line: 154, column: 21, scope: !1073)
!1087 = !DILocation(line: 154, column: 55, scope: !1073)
!1088 = !DILocation(line: 154, column: 53, scope: !1073)
!1089 = !DILocation(line: 154, column: 97, scope: !1073)
!1090 = !DILocation(line: 154, column: 37, scope: !1073)
!1091 = !DILocation(line: 156, column: 21, scope: !1073)
!1092 = !DILocation(line: 158, column: 78, scope: !1073)
!1093 = !DILocation(line: 158, column: 35, scope: !1073)
!1094 = !DILocation(line: 158, column: 33, scope: !1073)
!1095 = !DILocation(line: 159, column: 13, scope: !1073)
!1096 = !DILocation(line: 161, column: 21, scope: !1073)
!1097 = !DILocation(line: 161, column: 55, scope: !1073)
!1098 = !DILocation(line: 161, column: 53, scope: !1073)
!1099 = !DILocation(line: 161, column: 98, scope: !1073)
!1100 = !DILocation(line: 161, column: 37, scope: !1073)
!1101 = !DILocation(line: 163, column: 21, scope: !1073)
!1102 = !DILocation(line: 165, column: 79, scope: !1073)
!1103 = !DILocation(line: 165, column: 35, scope: !1073)
!1104 = !DILocation(line: 165, column: 33, scope: !1073)
!1105 = !DILocation(line: 167, column: 36, scope: !1073)
!1106 = !DILocation(line: 167, column: 44, scope: !1073)
!1107 = !DILocation(line: 167, column: 51, scope: !1073)
!1108 = !DILocation(line: 167, column: 41, scope: !1073)
!1109 = !DILocation(line: 167, column: 17, scope: !1073)
!1110 = !DILocation(line: 167, column: 34, scope: !1073)
!1111 = !DILocation(line: 168, column: 13, scope: !1073)
!1112 = !DILocation(line: 170, column: 13, scope: !1073)
!1113 = !DILocation(line: 171, column: 9, scope: !1073)
!1114 = !DILocation(line: 151, column: 43, scope: !1073)
!1115 = distinct !{!1115, !1081, !1113}
!1116 = !DILocation(line: 172, column: 13, scope: !1073)
!1117 = !DILocation(line: 172, column: 30, scope: !1073)
!1118 = !DILocation(line: 173, column: 16, scope: !1073)
!1119 = !DILocation(line: 173, column: 9, scope: !1073)
!1120 = distinct !DISubprogram(name: "requiredSpace4VecElem<long long, ON>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1121 = !DILocation(line: 103, column: 32, scope: !1120)
!1122 = !DILocation(line: 103, column: 25, scope: !1120)
!1123 = !DILocation(line: 103, column: 10, scope: !1120)
!1124 = !DILocation(line: 106, column: 16, scope: !1120)
!1125 = !DILocation(line: 106, column: 29, scope: !1120)
!1126 = !DILocation(line: 106, column: 9, scope: !1120)
!1127 = distinct !DISubprogram(name: "writeFormat4VecElem<long long, ON>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1128 = !DILocation(line: 212, column: 20, scope: !1127)
!1129 = !DILocation(line: 212, column: 14, scope: !1127)
!1130 = !DILocation(line: 215, column: 41, scope: !1127)
!1131 = !DILocation(line: 215, column: 51, scope: !1127)
!1132 = !DILocation(line: 215, column: 83, scope: !1127)
!1133 = !DILocation(line: 215, column: 81, scope: !1127)
!1134 = !DILocation(line: 215, column: 99, scope: !1127)
!1135 = !DILocation(line: 215, column: 28, scope: !1127)
!1136 = !DILocation(line: 215, column: 25, scope: !1127)
!1137 = !DILocation(line: 218, column: 16, scope: !1127)
!1138 = !DILocation(line: 218, column: 9, scope: !1127)
!1139 = distinct !DISubprogram(name: "requiredSpace4VecElem<long long, OFF>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1140 = !DILocation(line: 103, column: 32, scope: !1139)
!1141 = !DILocation(line: 103, column: 25, scope: !1139)
!1142 = !DILocation(line: 103, column: 10, scope: !1139)
!1143 = !DILocation(line: 109, column: 16, scope: !1139)
!1144 = !DILocation(line: 109, column: 29, scope: !1139)
!1145 = !DILocation(line: 109, column: 9, scope: !1139)
!1146 = distinct !DISubprogram(name: "writeFormat4VecElem<long long, OFF>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1147 = !DILocation(line: 212, column: 20, scope: !1146)
!1148 = !DILocation(line: 212, column: 14, scope: !1146)
!1149 = !DILocation(line: 214, column: 56, scope: !1146)
!1150 = !DILocation(line: 214, column: 88, scope: !1146)
!1151 = !DILocation(line: 214, column: 86, scope: !1146)
!1152 = !DILocation(line: 214, column: 104, scope: !1146)
!1153 = !DILocation(line: 214, column: 32, scope: !1146)
!1154 = !DILocation(line: 214, column: 29, scope: !1146)
!1155 = !DILocation(line: 215, column: 41, scope: !1146)
!1156 = !DILocation(line: 215, column: 51, scope: !1146)
!1157 = !DILocation(line: 215, column: 83, scope: !1146)
!1158 = !DILocation(line: 215, column: 81, scope: !1146)
!1159 = !DILocation(line: 215, column: 99, scope: !1146)
!1160 = !DILocation(line: 215, column: 28, scope: !1146)
!1161 = !DILocation(line: 215, column: 25, scope: !1146)
!1162 = !DILocation(line: 217, column: 56, scope: !1146)
!1163 = !DILocation(line: 217, column: 88, scope: !1146)
!1164 = !DILocation(line: 217, column: 86, scope: !1146)
!1165 = !DILocation(line: 217, column: 104, scope: !1146)
!1166 = !DILocation(line: 217, column: 32, scope: !1146)
!1167 = !DILocation(line: 217, column: 29, scope: !1146)
!1168 = !DILocation(line: 218, column: 16, scope: !1146)
!1169 = !DILocation(line: 218, column: 9, scope: !1146)
!1170 = distinct !DISubprogram(name: "UniArg2StrIfSuitable<unsigned long long, ArgWriter>", scope: !74, file: !74, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1171 = !DILocation(line: 82, column: 9, scope: !1170)
!1172 = !DILocation(line: 82, column: 17, scope: !1170)
!1173 = !DILocation(line: 82, column: 14, scope: !1170)
!1174 = !DILocation(line: 83, column: 15, scope: !1170)
!1175 = !DILocation(line: 83, column: 34, scope: !1170)
!1176 = !DILocation(line: 83, column: 13, scope: !1170)
!1177 = !DILocation(line: 84, column: 9, scope: !1170)
!1178 = !DILocation(line: 86, column: 5, scope: !1170)
!1179 = !DILocation(line: 87, column: 1, scope: !1170)
!1180 = distinct !DISubprogram(name: "VarArg2StrIfSuitable<unsigned long long, ArgWriter>", scope: !74, file: !74, line: 92, type: !16, scopeLine: 92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1181 = !DILocation(line: 93, column: 9, scope: !1180)
!1182 = !DILocation(line: 93, column: 17, scope: !1180)
!1183 = !DILocation(line: 93, column: 14, scope: !1180)
!1184 = !DILocation(line: 94, column: 15, scope: !1180)
!1185 = !DILocation(line: 94, column: 34, scope: !1180)
!1186 = !DILocation(line: 94, column: 13, scope: !1180)
!1187 = !DILocation(line: 95, column: 9, scope: !1180)
!1188 = !DILocation(line: 97, column: 5, scope: !1180)
!1189 = !DILocation(line: 98, column: 1, scope: !1180)
!1190 = distinct !DISubprogram(name: "getEncoding4Uniform<unsigned long long>", scope: !192, file: !192, line: 80, type: !16, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1191 = !DILocation(line: 80, column: 83, scope: !1190)
!1192 = distinct !DISubprogram(name: "uniform2Str<unsigned long long>", scope: !1, file: !1, line: 130, type: !16, scopeLine: 130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1193 = !DILocation(line: 131, column: 20, scope: !1192)
!1194 = !DILocation(line: 131, column: 14, scope: !1192)
!1195 = !DILocation(line: 133, column: 40, scope: !1192)
!1196 = !DILocation(line: 133, column: 27, scope: !1192)
!1197 = !DILocation(line: 133, column: 14, scope: !1192)
!1198 = !DILocation(line: 134, column: 13, scope: !1192)
!1199 = !DILocation(line: 134, column: 25, scope: !1192)
!1200 = !DILocation(line: 136, column: 9, scope: !1192)
!1201 = !DILocation(line: 137, column: 16, scope: !1192)
!1202 = !DILocation(line: 137, column: 9, scope: !1192)
!1203 = distinct !DISubprogram(name: "cmType2Specifier<unsigned long long>", scope: !1, file: !1, line: 90, type: !16, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1204 = !DILocation(line: 90, column: 70, scope: !1203)
!1205 = !DILocation(line: 90, column: 63, scope: !1203)
!1206 = distinct !DISubprogram(name: "WriteArg<unsigned long long>", scope: !1, file: !1, line: 181, type: !16, scopeLine: 181, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1207 = !DILocation(line: 184, column: 19, scope: !1206)
!1208 = !DILocation(line: 184, column: 17, scope: !1206)
!1209 = !DILocation(line: 185, column: 20, scope: !1206)
!1210 = !DILocation(line: 185, column: 18, scope: !1206)
!1211 = !DILocation(line: 190, column: 28, scope: !1206)
!1212 = !DILocation(line: 190, column: 22, scope: !1206)
!1213 = !DILocation(line: 191, column: 40, scope: !1206)
!1214 = !DILocation(line: 191, column: 45, scope: !1206)
!1215 = !DILocation(line: 191, column: 54, scope: !1206)
!1216 = !DILocation(line: 191, column: 59, scope: !1206)
!1217 = !DILocation(line: 191, column: 9, scope: !1206)
!1218 = !DILocation(line: 192, column: 9, scope: !1206)
!1219 = !DILocation(line: 192, column: 17, scope: !1206)
!1220 = !DILocation(line: 193, column: 5, scope: !1206)
!1221 = distinct !DISubprogram(name: "type2Specifier<unsigned long long>", scope: !192, file: !192, line: 123, type: !16, scopeLine: 123, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1222 = !DILocation(line: 123, column: 71, scope: !1221)
!1223 = distinct !DISubprogram(name: "getEncoding4Varying<unsigned long long>", scope: !192, file: !192, line: 89, type: !16, scopeLine: 89, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1224 = !DILocation(line: 89, column: 83, scope: !1223)
!1225 = distinct !DISubprogram(name: "varying2Str<unsigned long long>", scope: !1, file: !1, line: 147, type: !16, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1226 = !DILocation(line: 149, column: 16, scope: !1225)
!1227 = !DILocation(line: 150, column: 13, scope: !1225)
!1228 = !DILocation(line: 151, column: 18, scope: !1225)
!1229 = !DILocation(line: 151, column: 14, scope: !1225)
!1230 = !DILocation(line: 151, column: 28, scope: !1225)
!1231 = !DILocation(line: 151, column: 35, scope: !1225)
!1232 = !DILocation(line: 151, column: 33, scope: !1225)
!1233 = !DILocation(line: 151, column: 9, scope: !1225)
!1234 = !DILocation(line: 152, column: 17, scope: !1225)
!1235 = !DILocation(line: 152, column: 34, scope: !1225)
!1236 = !DILocation(line: 152, column: 31, scope: !1225)
!1237 = !DILocation(line: 152, column: 23, scope: !1225)
!1238 = !DILocation(line: 154, column: 21, scope: !1225)
!1239 = !DILocation(line: 154, column: 55, scope: !1225)
!1240 = !DILocation(line: 154, column: 53, scope: !1225)
!1241 = !DILocation(line: 154, column: 97, scope: !1225)
!1242 = !DILocation(line: 154, column: 37, scope: !1225)
!1243 = !DILocation(line: 156, column: 21, scope: !1225)
!1244 = !DILocation(line: 158, column: 78, scope: !1225)
!1245 = !DILocation(line: 158, column: 35, scope: !1225)
!1246 = !DILocation(line: 158, column: 33, scope: !1225)
!1247 = !DILocation(line: 159, column: 13, scope: !1225)
!1248 = !DILocation(line: 161, column: 21, scope: !1225)
!1249 = !DILocation(line: 161, column: 55, scope: !1225)
!1250 = !DILocation(line: 161, column: 53, scope: !1225)
!1251 = !DILocation(line: 161, column: 98, scope: !1225)
!1252 = !DILocation(line: 161, column: 37, scope: !1225)
!1253 = !DILocation(line: 163, column: 21, scope: !1225)
!1254 = !DILocation(line: 165, column: 79, scope: !1225)
!1255 = !DILocation(line: 165, column: 35, scope: !1225)
!1256 = !DILocation(line: 165, column: 33, scope: !1225)
!1257 = !DILocation(line: 167, column: 36, scope: !1225)
!1258 = !DILocation(line: 167, column: 44, scope: !1225)
!1259 = !DILocation(line: 167, column: 51, scope: !1225)
!1260 = !DILocation(line: 167, column: 41, scope: !1225)
!1261 = !DILocation(line: 167, column: 17, scope: !1225)
!1262 = !DILocation(line: 167, column: 34, scope: !1225)
!1263 = !DILocation(line: 168, column: 13, scope: !1225)
!1264 = !DILocation(line: 170, column: 13, scope: !1225)
!1265 = !DILocation(line: 171, column: 9, scope: !1225)
!1266 = !DILocation(line: 151, column: 43, scope: !1225)
!1267 = distinct !{!1267, !1233, !1265}
!1268 = !DILocation(line: 172, column: 13, scope: !1225)
!1269 = !DILocation(line: 172, column: 30, scope: !1225)
!1270 = !DILocation(line: 173, column: 16, scope: !1225)
!1271 = !DILocation(line: 173, column: 9, scope: !1225)
!1272 = distinct !DISubprogram(name: "requiredSpace4VecElem<unsigned long long, ON>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1273 = !DILocation(line: 103, column: 32, scope: !1272)
!1274 = !DILocation(line: 103, column: 25, scope: !1272)
!1275 = !DILocation(line: 103, column: 10, scope: !1272)
!1276 = !DILocation(line: 106, column: 16, scope: !1272)
!1277 = !DILocation(line: 106, column: 29, scope: !1272)
!1278 = !DILocation(line: 106, column: 9, scope: !1272)
!1279 = distinct !DISubprogram(name: "writeFormat4VecElem<unsigned long long, ON>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1280 = !DILocation(line: 212, column: 20, scope: !1279)
!1281 = !DILocation(line: 212, column: 14, scope: !1279)
!1282 = !DILocation(line: 215, column: 41, scope: !1279)
!1283 = !DILocation(line: 215, column: 51, scope: !1279)
!1284 = !DILocation(line: 215, column: 83, scope: !1279)
!1285 = !DILocation(line: 215, column: 81, scope: !1279)
!1286 = !DILocation(line: 215, column: 99, scope: !1279)
!1287 = !DILocation(line: 215, column: 28, scope: !1279)
!1288 = !DILocation(line: 215, column: 25, scope: !1279)
!1289 = !DILocation(line: 218, column: 16, scope: !1279)
!1290 = !DILocation(line: 218, column: 9, scope: !1279)
!1291 = distinct !DISubprogram(name: "requiredSpace4VecElem<unsigned long long, OFF>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1292 = !DILocation(line: 103, column: 32, scope: !1291)
!1293 = !DILocation(line: 103, column: 25, scope: !1291)
!1294 = !DILocation(line: 103, column: 10, scope: !1291)
!1295 = !DILocation(line: 109, column: 16, scope: !1291)
!1296 = !DILocation(line: 109, column: 29, scope: !1291)
!1297 = !DILocation(line: 109, column: 9, scope: !1291)
!1298 = distinct !DISubprogram(name: "writeFormat4VecElem<unsigned long long, OFF>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1299 = !DILocation(line: 212, column: 20, scope: !1298)
!1300 = !DILocation(line: 212, column: 14, scope: !1298)
!1301 = !DILocation(line: 214, column: 56, scope: !1298)
!1302 = !DILocation(line: 214, column: 88, scope: !1298)
!1303 = !DILocation(line: 214, column: 86, scope: !1298)
!1304 = !DILocation(line: 214, column: 104, scope: !1298)
!1305 = !DILocation(line: 214, column: 32, scope: !1298)
!1306 = !DILocation(line: 214, column: 29, scope: !1298)
!1307 = !DILocation(line: 215, column: 41, scope: !1298)
!1308 = !DILocation(line: 215, column: 51, scope: !1298)
!1309 = !DILocation(line: 215, column: 83, scope: !1298)
!1310 = !DILocation(line: 215, column: 81, scope: !1298)
!1311 = !DILocation(line: 215, column: 99, scope: !1298)
!1312 = !DILocation(line: 215, column: 28, scope: !1298)
!1313 = !DILocation(line: 215, column: 25, scope: !1298)
!1314 = !DILocation(line: 217, column: 56, scope: !1298)
!1315 = !DILocation(line: 217, column: 88, scope: !1298)
!1316 = !DILocation(line: 217, column: 86, scope: !1298)
!1317 = !DILocation(line: 217, column: 104, scope: !1298)
!1318 = !DILocation(line: 217, column: 32, scope: !1298)
!1319 = !DILocation(line: 217, column: 29, scope: !1298)
!1320 = !DILocation(line: 218, column: 16, scope: !1298)
!1321 = !DILocation(line: 218, column: 9, scope: !1298)
!1322 = distinct !DISubprogram(name: "UniArg2StrIfSuitable<double, ArgWriter>", scope: !74, file: !74, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1323 = !DILocation(line: 82, column: 9, scope: !1322)
!1324 = !DILocation(line: 82, column: 17, scope: !1322)
!1325 = !DILocation(line: 82, column: 14, scope: !1322)
!1326 = !DILocation(line: 83, column: 15, scope: !1322)
!1327 = !DILocation(line: 83, column: 34, scope: !1322)
!1328 = !DILocation(line: 83, column: 13, scope: !1322)
!1329 = !DILocation(line: 84, column: 9, scope: !1322)
!1330 = !DILocation(line: 86, column: 5, scope: !1322)
!1331 = !DILocation(line: 87, column: 1, scope: !1322)
!1332 = distinct !DISubprogram(name: "VarArg2StrIfSuitable<double, ArgWriter>", scope: !74, file: !74, line: 92, type: !16, scopeLine: 92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1333 = !DILocation(line: 93, column: 9, scope: !1332)
!1334 = !DILocation(line: 93, column: 17, scope: !1332)
!1335 = !DILocation(line: 93, column: 14, scope: !1332)
!1336 = !DILocation(line: 94, column: 15, scope: !1332)
!1337 = !DILocation(line: 94, column: 34, scope: !1332)
!1338 = !DILocation(line: 94, column: 13, scope: !1332)
!1339 = !DILocation(line: 95, column: 9, scope: !1332)
!1340 = !DILocation(line: 97, column: 5, scope: !1332)
!1341 = !DILocation(line: 98, column: 1, scope: !1332)
!1342 = distinct !DISubprogram(name: "getEncoding4Uniform<double>", scope: !192, file: !192, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1343 = !DILocation(line: 81, column: 71, scope: !1342)
!1344 = distinct !DISubprogram(name: "uniform2Str<double>", scope: !1, file: !1, line: 130, type: !16, scopeLine: 130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1345 = !DILocation(line: 131, column: 20, scope: !1344)
!1346 = !DILocation(line: 131, column: 14, scope: !1344)
!1347 = !DILocation(line: 133, column: 40, scope: !1344)
!1348 = !DILocation(line: 133, column: 27, scope: !1344)
!1349 = !DILocation(line: 133, column: 14, scope: !1344)
!1350 = !DILocation(line: 134, column: 13, scope: !1344)
!1351 = !DILocation(line: 134, column: 25, scope: !1344)
!1352 = !DILocation(line: 136, column: 9, scope: !1344)
!1353 = !DILocation(line: 137, column: 16, scope: !1344)
!1354 = !DILocation(line: 137, column: 9, scope: !1344)
!1355 = distinct !DISubprogram(name: "cmType2Specifier<double>", scope: !1, file: !1, line: 90, type: !16, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1356 = !DILocation(line: 90, column: 70, scope: !1355)
!1357 = !DILocation(line: 90, column: 63, scope: !1355)
!1358 = distinct !DISubprogram(name: "WriteArg<double>", scope: !1, file: !1, line: 181, type: !16, scopeLine: 181, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1359 = !DILocation(line: 184, column: 19, scope: !1358)
!1360 = !DILocation(line: 184, column: 17, scope: !1358)
!1361 = !DILocation(line: 185, column: 20, scope: !1358)
!1362 = !DILocation(line: 185, column: 18, scope: !1358)
!1363 = !DILocation(line: 190, column: 28, scope: !1358)
!1364 = !DILocation(line: 190, column: 22, scope: !1358)
!1365 = !DILocation(line: 191, column: 40, scope: !1358)
!1366 = !DILocation(line: 191, column: 45, scope: !1358)
!1367 = !DILocation(line: 191, column: 54, scope: !1358)
!1368 = !DILocation(line: 191, column: 59, scope: !1358)
!1369 = !DILocation(line: 191, column: 9, scope: !1358)
!1370 = !DILocation(line: 192, column: 9, scope: !1358)
!1371 = !DILocation(line: 192, column: 17, scope: !1358)
!1372 = !DILocation(line: 193, column: 5, scope: !1358)
!1373 = distinct !DISubprogram(name: "type2Specifier<double>", scope: !192, file: !192, line: 124, type: !16, scopeLine: 124, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1374 = !DILocation(line: 124, column: 59, scope: !1373)
!1375 = distinct !DISubprogram(name: "getEncoding4Varying<double>", scope: !192, file: !192, line: 90, type: !16, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1376 = !DILocation(line: 90, column: 71, scope: !1375)
!1377 = distinct !DISubprogram(name: "varying2Str<double>", scope: !1, file: !1, line: 147, type: !16, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1378 = !DILocation(line: 149, column: 16, scope: !1377)
!1379 = !DILocation(line: 150, column: 13, scope: !1377)
!1380 = !DILocation(line: 151, column: 18, scope: !1377)
!1381 = !DILocation(line: 151, column: 14, scope: !1377)
!1382 = !DILocation(line: 151, column: 28, scope: !1377)
!1383 = !DILocation(line: 151, column: 35, scope: !1377)
!1384 = !DILocation(line: 151, column: 33, scope: !1377)
!1385 = !DILocation(line: 151, column: 9, scope: !1377)
!1386 = !DILocation(line: 152, column: 17, scope: !1377)
!1387 = !DILocation(line: 152, column: 34, scope: !1377)
!1388 = !DILocation(line: 152, column: 31, scope: !1377)
!1389 = !DILocation(line: 152, column: 23, scope: !1377)
!1390 = !DILocation(line: 154, column: 21, scope: !1377)
!1391 = !DILocation(line: 154, column: 55, scope: !1377)
!1392 = !DILocation(line: 154, column: 53, scope: !1377)
!1393 = !DILocation(line: 154, column: 97, scope: !1377)
!1394 = !DILocation(line: 154, column: 37, scope: !1377)
!1395 = !DILocation(line: 156, column: 21, scope: !1377)
!1396 = !DILocation(line: 158, column: 78, scope: !1377)
!1397 = !DILocation(line: 158, column: 35, scope: !1377)
!1398 = !DILocation(line: 158, column: 33, scope: !1377)
!1399 = !DILocation(line: 159, column: 13, scope: !1377)
!1400 = !DILocation(line: 161, column: 21, scope: !1377)
!1401 = !DILocation(line: 161, column: 55, scope: !1377)
!1402 = !DILocation(line: 161, column: 53, scope: !1377)
!1403 = !DILocation(line: 161, column: 98, scope: !1377)
!1404 = !DILocation(line: 161, column: 37, scope: !1377)
!1405 = !DILocation(line: 163, column: 21, scope: !1377)
!1406 = !DILocation(line: 165, column: 79, scope: !1377)
!1407 = !DILocation(line: 165, column: 35, scope: !1377)
!1408 = !DILocation(line: 165, column: 33, scope: !1377)
!1409 = !DILocation(line: 167, column: 36, scope: !1377)
!1410 = !DILocation(line: 167, column: 44, scope: !1377)
!1411 = !DILocation(line: 167, column: 51, scope: !1377)
!1412 = !DILocation(line: 167, column: 41, scope: !1377)
!1413 = !DILocation(line: 167, column: 17, scope: !1377)
!1414 = !DILocation(line: 167, column: 34, scope: !1377)
!1415 = !DILocation(line: 168, column: 13, scope: !1377)
!1416 = !DILocation(line: 170, column: 13, scope: !1377)
!1417 = !DILocation(line: 171, column: 9, scope: !1377)
!1418 = !DILocation(line: 151, column: 43, scope: !1377)
!1419 = distinct !{!1419, !1385, !1417}
!1420 = !DILocation(line: 172, column: 13, scope: !1377)
!1421 = !DILocation(line: 172, column: 30, scope: !1377)
!1422 = !DILocation(line: 173, column: 16, scope: !1377)
!1423 = !DILocation(line: 173, column: 9, scope: !1377)
!1424 = distinct !DISubprogram(name: "requiredSpace4VecElem<double, ON>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1425 = !DILocation(line: 103, column: 32, scope: !1424)
!1426 = !DILocation(line: 103, column: 25, scope: !1424)
!1427 = !DILocation(line: 103, column: 10, scope: !1424)
!1428 = !DILocation(line: 106, column: 16, scope: !1424)
!1429 = !DILocation(line: 106, column: 29, scope: !1424)
!1430 = !DILocation(line: 106, column: 9, scope: !1424)
!1431 = distinct !DISubprogram(name: "writeFormat4VecElem<double, ON>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1432 = !DILocation(line: 212, column: 20, scope: !1431)
!1433 = !DILocation(line: 212, column: 14, scope: !1431)
!1434 = !DILocation(line: 215, column: 41, scope: !1431)
!1435 = !DILocation(line: 215, column: 51, scope: !1431)
!1436 = !DILocation(line: 215, column: 83, scope: !1431)
!1437 = !DILocation(line: 215, column: 81, scope: !1431)
!1438 = !DILocation(line: 215, column: 99, scope: !1431)
!1439 = !DILocation(line: 215, column: 28, scope: !1431)
!1440 = !DILocation(line: 215, column: 25, scope: !1431)
!1441 = !DILocation(line: 218, column: 16, scope: !1431)
!1442 = !DILocation(line: 218, column: 9, scope: !1431)
!1443 = distinct !DISubprogram(name: "requiredSpace4VecElem<double, OFF>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1444 = !DILocation(line: 103, column: 32, scope: !1443)
!1445 = !DILocation(line: 103, column: 25, scope: !1443)
!1446 = !DILocation(line: 103, column: 10, scope: !1443)
!1447 = !DILocation(line: 109, column: 16, scope: !1443)
!1448 = !DILocation(line: 109, column: 29, scope: !1443)
!1449 = !DILocation(line: 109, column: 9, scope: !1443)
!1450 = distinct !DISubprogram(name: "writeFormat4VecElem<double, OFF>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1451 = !DILocation(line: 212, column: 20, scope: !1450)
!1452 = !DILocation(line: 212, column: 14, scope: !1450)
!1453 = !DILocation(line: 214, column: 56, scope: !1450)
!1454 = !DILocation(line: 214, column: 88, scope: !1450)
!1455 = !DILocation(line: 214, column: 86, scope: !1450)
!1456 = !DILocation(line: 214, column: 104, scope: !1450)
!1457 = !DILocation(line: 214, column: 32, scope: !1450)
!1458 = !DILocation(line: 214, column: 29, scope: !1450)
!1459 = !DILocation(line: 215, column: 41, scope: !1450)
!1460 = !DILocation(line: 215, column: 51, scope: !1450)
!1461 = !DILocation(line: 215, column: 83, scope: !1450)
!1462 = !DILocation(line: 215, column: 81, scope: !1450)
!1463 = !DILocation(line: 215, column: 99, scope: !1450)
!1464 = !DILocation(line: 215, column: 28, scope: !1450)
!1465 = !DILocation(line: 215, column: 25, scope: !1450)
!1466 = !DILocation(line: 217, column: 56, scope: !1450)
!1467 = !DILocation(line: 217, column: 88, scope: !1450)
!1468 = !DILocation(line: 217, column: 86, scope: !1450)
!1469 = !DILocation(line: 217, column: 104, scope: !1450)
!1470 = !DILocation(line: 217, column: 32, scope: !1450)
!1471 = !DILocation(line: 217, column: 29, scope: !1450)
!1472 = !DILocation(line: 218, column: 16, scope: !1450)
!1473 = !DILocation(line: 218, column: 9, scope: !1450)
!1474 = distinct !DISubprogram(name: "UniArg2StrIfSuitable<void *, ArgWriter>", scope: !74, file: !74, line: 81, type: !16, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1475 = !DILocation(line: 82, column: 9, scope: !1474)
!1476 = !DILocation(line: 82, column: 17, scope: !1474)
!1477 = !DILocation(line: 82, column: 14, scope: !1474)
!1478 = !DILocation(line: 83, column: 15, scope: !1474)
!1479 = !DILocation(line: 83, column: 34, scope: !1474)
!1480 = !DILocation(line: 83, column: 13, scope: !1474)
!1481 = !DILocation(line: 84, column: 9, scope: !1474)
!1482 = !DILocation(line: 86, column: 5, scope: !1474)
!1483 = !DILocation(line: 87, column: 1, scope: !1474)
!1484 = distinct !DISubprogram(name: "VarArg2StrIfSuitable<void *, ArgWriter>", scope: !74, file: !74, line: 92, type: !16, scopeLine: 92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1485 = !DILocation(line: 93, column: 9, scope: !1484)
!1486 = !DILocation(line: 93, column: 17, scope: !1484)
!1487 = !DILocation(line: 93, column: 14, scope: !1484)
!1488 = !DILocation(line: 94, column: 15, scope: !1484)
!1489 = !DILocation(line: 94, column: 34, scope: !1484)
!1490 = !DILocation(line: 94, column: 13, scope: !1484)
!1491 = !DILocation(line: 95, column: 9, scope: !1484)
!1492 = !DILocation(line: 97, column: 5, scope: !1484)
!1493 = !DILocation(line: 98, column: 1, scope: !1484)
!1494 = distinct !DISubprogram(name: "getEncoding4Uniform<void *>", scope: !192, file: !192, line: 82, type: !16, scopeLine: 82, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1495 = !DILocation(line: 82, column: 71, scope: !1494)
!1496 = distinct !DISubprogram(name: "uniform2Str<void *>", scope: !1, file: !1, line: 130, type: !16, scopeLine: 130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1497 = !DILocation(line: 131, column: 20, scope: !1496)
!1498 = !DILocation(line: 131, column: 14, scope: !1496)
!1499 = !DILocation(line: 133, column: 40, scope: !1496)
!1500 = !DILocation(line: 133, column: 27, scope: !1496)
!1501 = !DILocation(line: 133, column: 14, scope: !1496)
!1502 = !DILocation(line: 134, column: 13, scope: !1496)
!1503 = !DILocation(line: 134, column: 25, scope: !1496)
!1504 = !DILocation(line: 136, column: 9, scope: !1496)
!1505 = !DILocation(line: 137, column: 16, scope: !1496)
!1506 = !DILocation(line: 137, column: 9, scope: !1496)
!1507 = distinct !DISubprogram(name: "cmType2Specifier<void *>", scope: !1, file: !1, line: 93, type: !16, scopeLine: 93, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1508 = !DILocation(line: 97, column: 9, scope: !1507)
!1509 = distinct !DISubprogram(name: "WriteArg<void *>", scope: !1, file: !1, line: 197, type: !16, scopeLine: 197, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1510 = !DILocation(line: 201, column: 13, scope: !1509)
!1511 = !DILocation(line: 202, column: 5, scope: !1509)
!1512 = distinct !DISubprogram(name: "getEncoding4Varying<void *>", scope: !192, file: !192, line: 91, type: !16, scopeLine: 91, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1513 = !DILocation(line: 91, column: 71, scope: !1512)
!1514 = distinct !DISubprogram(name: "varying2Str<void *>", scope: !1, file: !1, line: 147, type: !16, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1515 = !DILocation(line: 149, column: 16, scope: !1514)
!1516 = !DILocation(line: 150, column: 13, scope: !1514)
!1517 = !DILocation(line: 151, column: 18, scope: !1514)
!1518 = !DILocation(line: 151, column: 14, scope: !1514)
!1519 = !DILocation(line: 151, column: 28, scope: !1514)
!1520 = !DILocation(line: 151, column: 35, scope: !1514)
!1521 = !DILocation(line: 151, column: 33, scope: !1514)
!1522 = !DILocation(line: 151, column: 9, scope: !1514)
!1523 = !DILocation(line: 152, column: 17, scope: !1514)
!1524 = !DILocation(line: 152, column: 34, scope: !1514)
!1525 = !DILocation(line: 152, column: 31, scope: !1514)
!1526 = !DILocation(line: 152, column: 23, scope: !1514)
!1527 = !DILocation(line: 154, column: 21, scope: !1514)
!1528 = !DILocation(line: 154, column: 55, scope: !1514)
!1529 = !DILocation(line: 154, column: 53, scope: !1514)
!1530 = !DILocation(line: 154, column: 97, scope: !1514)
!1531 = !DILocation(line: 154, column: 37, scope: !1514)
!1532 = !DILocation(line: 156, column: 21, scope: !1514)
!1533 = !DILocation(line: 158, column: 78, scope: !1514)
!1534 = !DILocation(line: 158, column: 35, scope: !1514)
!1535 = !DILocation(line: 158, column: 33, scope: !1514)
!1536 = !DILocation(line: 159, column: 13, scope: !1514)
!1537 = !DILocation(line: 161, column: 21, scope: !1514)
!1538 = !DILocation(line: 161, column: 55, scope: !1514)
!1539 = !DILocation(line: 161, column: 53, scope: !1514)
!1540 = !DILocation(line: 161, column: 98, scope: !1514)
!1541 = !DILocation(line: 161, column: 37, scope: !1514)
!1542 = !DILocation(line: 163, column: 21, scope: !1514)
!1543 = !DILocation(line: 165, column: 79, scope: !1514)
!1544 = !DILocation(line: 165, column: 35, scope: !1514)
!1545 = !DILocation(line: 165, column: 33, scope: !1514)
!1546 = !DILocation(line: 167, column: 36, scope: !1514)
!1547 = !DILocation(line: 167, column: 44, scope: !1514)
!1548 = !DILocation(line: 167, column: 51, scope: !1514)
!1549 = !DILocation(line: 167, column: 41, scope: !1514)
!1550 = !DILocation(line: 167, column: 17, scope: !1514)
!1551 = !DILocation(line: 167, column: 34, scope: !1514)
!1552 = !DILocation(line: 168, column: 13, scope: !1514)
!1553 = !DILocation(line: 170, column: 13, scope: !1514)
!1554 = !DILocation(line: 171, column: 9, scope: !1514)
!1555 = !DILocation(line: 151, column: 43, scope: !1514)
!1556 = distinct !{!1556, !1522, !1554}
!1557 = !DILocation(line: 172, column: 13, scope: !1514)
!1558 = !DILocation(line: 172, column: 30, scope: !1514)
!1559 = !DILocation(line: 173, column: 16, scope: !1514)
!1560 = !DILocation(line: 173, column: 9, scope: !1514)
!1561 = distinct !DISubprogram(name: "requiredSpace4VecElem<void *, ON>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1562 = !DILocation(line: 103, column: 32, scope: !1561)
!1563 = !DILocation(line: 103, column: 25, scope: !1561)
!1564 = !DILocation(line: 103, column: 10, scope: !1561)
!1565 = !DILocation(line: 106, column: 16, scope: !1561)
!1566 = !DILocation(line: 106, column: 29, scope: !1561)
!1567 = !DILocation(line: 106, column: 9, scope: !1561)
!1568 = distinct !DISubprogram(name: "writeFormat4VecElem<void *, ON>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1569 = !DILocation(line: 212, column: 20, scope: !1568)
!1570 = !DILocation(line: 212, column: 14, scope: !1568)
!1571 = !DILocation(line: 215, column: 41, scope: !1568)
!1572 = !DILocation(line: 215, column: 51, scope: !1568)
!1573 = !DILocation(line: 215, column: 83, scope: !1568)
!1574 = !DILocation(line: 215, column: 81, scope: !1568)
!1575 = !DILocation(line: 215, column: 99, scope: !1568)
!1576 = !DILocation(line: 215, column: 28, scope: !1568)
!1577 = !DILocation(line: 215, column: 25, scope: !1568)
!1578 = !DILocation(line: 218, column: 16, scope: !1568)
!1579 = !DILocation(line: 218, column: 9, scope: !1568)
!1580 = distinct !DISubprogram(name: "requiredSpace4VecElem<void *, OFF>", scope: !1, file: !1, line: 102, type: !16, scopeLine: 102, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1581 = !DILocation(line: 103, column: 32, scope: !1580)
!1582 = !DILocation(line: 103, column: 25, scope: !1580)
!1583 = !DILocation(line: 103, column: 10, scope: !1580)
!1584 = !DILocation(line: 109, column: 16, scope: !1580)
!1585 = !DILocation(line: 109, column: 29, scope: !1580)
!1586 = !DILocation(line: 109, column: 9, scope: !1580)
!1587 = distinct !DISubprogram(name: "writeFormat4VecElem<void *, OFF>", scope: !1, file: !1, line: 211, type: !16, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1588 = !DILocation(line: 212, column: 20, scope: !1587)
!1589 = !DILocation(line: 212, column: 14, scope: !1587)
!1590 = !DILocation(line: 214, column: 56, scope: !1587)
!1591 = !DILocation(line: 214, column: 88, scope: !1587)
!1592 = !DILocation(line: 214, column: 86, scope: !1587)
!1593 = !DILocation(line: 214, column: 104, scope: !1587)
!1594 = !DILocation(line: 214, column: 32, scope: !1587)
!1595 = !DILocation(line: 214, column: 29, scope: !1587)
!1596 = !DILocation(line: 215, column: 41, scope: !1587)
!1597 = !DILocation(line: 215, column: 51, scope: !1587)
!1598 = !DILocation(line: 215, column: 83, scope: !1587)
!1599 = !DILocation(line: 215, column: 81, scope: !1587)
!1600 = !DILocation(line: 215, column: 99, scope: !1587)
!1601 = !DILocation(line: 215, column: 28, scope: !1587)
!1602 = !DILocation(line: 215, column: 25, scope: !1587)
!1603 = !DILocation(line: 217, column: 56, scope: !1587)
!1604 = !DILocation(line: 217, column: 88, scope: !1587)
!1605 = !DILocation(line: 217, column: 86, scope: !1587)
!1606 = !DILocation(line: 217, column: 104, scope: !1587)
!1607 = !DILocation(line: 217, column: 32, scope: !1587)
!1608 = !DILocation(line: 217, column: 29, scope: !1587)
!1609 = !DILocation(line: 218, column: 16, scope: !1587)
!1610 = !DILocation(line: 218, column: 9, scope: !1587)
!1611 = distinct !DISubprogram(name: "CopyTillSep<'\5Cx00', 100, 128>", scope: !74, file: !74, line: 68, type: !16, scopeLine: 69, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1612 = !DILocation(line: 70, column: 29, scope: !1611)
!1613 = !DILocation(line: 70, column: 16, scope: !1611)
!1614 = !DILocation(line: 71, column: 5, scope: !1611)
!1615 = !DILocation(line: 71, column: 18, scope: !1611)
!1616 = !DILocation(line: 71, column: 14, scope: !1611)
!1617 = !DILocation(line: 71, column: 26, scope: !1611)
!1618 = !DILocation(line: 71, column: 43, scope: !1611)
!1619 = !DILocation(line: 71, column: 46, scope: !1611)
!1620 = !DILocation(line: 0, scope: !1611)
!1621 = !DILocation(line: 72, column: 35, scope: !1611)
!1622 = !DILocation(line: 72, column: 29, scope: !1611)
!1623 = !DILocation(line: 72, column: 25, scope: !1611)
!1624 = !DILocation(line: 72, column: 19, scope: !1611)
!1625 = !DILocation(line: 72, column: 13, scope: !1611)
!1626 = !DILocation(line: 72, column: 23, scope: !1611)
!1627 = !DILocation(line: 73, column: 9, scope: !1611)
!1628 = distinct !{!1628, !1614, !1629}
!1629 = !DILocation(line: 74, column: 5, scope: !1611)
!1630 = !DILocation(line: 75, column: 12, scope: !1611)
!1631 = !DILocation(line: 75, column: 21, scope: !1611)
!1632 = !DILocation(line: 75, column: 19, scope: !1611)
!1633 = !DILocation(line: 75, column: 5, scope: !1611)
!1634 = distinct !DISubprogram(name: "vector_cast<1, unsigned int>", scope: !1, file: !1, line: 291, type: !16, scopeLine: 291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1635 = !DILocation(line: 292, column: 28, scope: !1634)
!1636 = !DILocation(line: 292, column: 22, scope: !1634)
!1637 = !DILocation(line: 293, column: 12, scope: !1634)
!1638 = !DILocation(line: 293, column: 5, scope: !1634)
!1639 = distinct !DISubprogram(name: "vector_cast<1, int>", scope: !1, file: !1, line: 291, type: !16, scopeLine: 291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1640 = !DILocation(line: 292, column: 28, scope: !1639)
!1641 = !DILocation(line: 292, column: 22, scope: !1639)
!1642 = !DILocation(line: 293, column: 12, scope: !1639)
!1643 = !DILocation(line: 293, column: 5, scope: !1639)
!1644 = !{i32 7028}
!1645 = distinct !DISubprogram(name: "switchEncoding4Uniform<UniformWriter>", scope: !192, file: !192, line: 158, type: !16, scopeLine: 158, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1646 = !DILocation(line: 159, column: 35, scope: !1645)
!1647 = !DILocation(line: 159, column: 41, scope: !1645)
!1648 = !DILocation(line: 159, column: 12, scope: !1645)
!1649 = !DILocation(line: 159, column: 5, scope: !1645)
!1650 = distinct !DISubprogram(name: "switchEncoding4Varying<VaryingWriter>", scope: !192, file: !192, line: 162, type: !16, scopeLine: 162, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1651 = !DILocation(line: 163, column: 35, scope: !1650)
!1652 = !DILocation(line: 163, column: 41, scope: !1650)
!1653 = !DILocation(line: 163, column: 12, scope: !1650)
!1654 = !DILocation(line: 163, column: 5, scope: !1650)
!1655 = distinct !DISubprogram(name: "switchEncoding<UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 143, type: !16, scopeLine: 143, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1656 = !DILocation(line: 144, column: 40, scope: !1655)
!1657 = !DILocation(line: 144, column: 46, scope: !1655)
!1658 = !DILocation(line: 144, column: 12, scope: !1655)
!1659 = !DILocation(line: 144, column: 58, scope: !1655)
!1660 = !DILocation(line: 144, column: 88, scope: !1655)
!1661 = !DILocation(line: 144, column: 94, scope: !1655)
!1662 = !DILocation(line: 144, column: 61, scope: !1655)
!1663 = !DILocation(line: 144, column: 106, scope: !1655)
!1664 = !DILocation(line: 145, column: 44, scope: !1655)
!1665 = !DILocation(line: 145, column: 50, scope: !1655)
!1666 = !DILocation(line: 145, column: 12, scope: !1655)
!1667 = !DILocation(line: 145, column: 62, scope: !1655)
!1668 = !DILocation(line: 145, column: 94, scope: !1655)
!1669 = !DILocation(line: 145, column: 100, scope: !1655)
!1670 = !DILocation(line: 145, column: 65, scope: !1655)
!1671 = !DILocation(line: 145, column: 112, scope: !1655)
!1672 = !DILocation(line: 146, column: 45, scope: !1655)
!1673 = !DILocation(line: 146, column: 51, scope: !1655)
!1674 = !DILocation(line: 146, column: 12, scope: !1655)
!1675 = !DILocation(line: 146, column: 63, scope: !1655)
!1676 = !DILocation(line: 147, column: 54, scope: !1655)
!1677 = !DILocation(line: 147, column: 60, scope: !1655)
!1678 = !DILocation(line: 147, column: 12, scope: !1655)
!1679 = !DILocation(line: 147, column: 72, scope: !1655)
!1680 = !DILocation(line: 148, column: 42, scope: !1655)
!1681 = !DILocation(line: 148, column: 48, scope: !1655)
!1682 = !DILocation(line: 148, column: 12, scope: !1655)
!1683 = !DILocation(line: 148, column: 60, scope: !1655)
!1684 = !DILocation(line: 148, column: 93, scope: !1655)
!1685 = !DILocation(line: 148, column: 99, scope: !1655)
!1686 = !DILocation(line: 148, column: 63, scope: !1655)
!1687 = !DILocation(line: 144, column: 5, scope: !1655)
!1688 = distinct !DISubprogram(name: "applyIfProperEncoding<bool, UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1689 = !DILocation(line: 136, column: 9, scope: !1688)
!1690 = !DILocation(line: 136, column: 34, scope: !1688)
!1691 = !DILocation(line: 136, column: 17, scope: !1688)
!1692 = !DILocation(line: 136, column: 14, scope: !1688)
!1693 = !DILocation(line: 137, column: 20, scope: !1688)
!1694 = !DILocation(line: 138, column: 9, scope: !1688)
!1695 = !DILocation(line: 140, column: 5, scope: !1688)
!1696 = !DILocation(line: 141, column: 1, scope: !1688)
!1697 = distinct !DISubprogram(name: "applyIfProperEncoding<int, UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1698 = !DILocation(line: 136, column: 9, scope: !1697)
!1699 = !DILocation(line: 136, column: 34, scope: !1697)
!1700 = !DILocation(line: 136, column: 17, scope: !1697)
!1701 = !DILocation(line: 136, column: 14, scope: !1697)
!1702 = !DILocation(line: 137, column: 20, scope: !1697)
!1703 = !DILocation(line: 138, column: 9, scope: !1697)
!1704 = !DILocation(line: 140, column: 5, scope: !1697)
!1705 = !DILocation(line: 141, column: 1, scope: !1697)
!1706 = distinct !DISubprogram(name: "applyIfProperEncoding<unsigned int, UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1707 = !DILocation(line: 136, column: 9, scope: !1706)
!1708 = !DILocation(line: 136, column: 34, scope: !1706)
!1709 = !DILocation(line: 136, column: 17, scope: !1706)
!1710 = !DILocation(line: 136, column: 14, scope: !1706)
!1711 = !DILocation(line: 137, column: 20, scope: !1706)
!1712 = !DILocation(line: 138, column: 9, scope: !1706)
!1713 = !DILocation(line: 140, column: 5, scope: !1706)
!1714 = !DILocation(line: 141, column: 1, scope: !1706)
!1715 = distinct !DISubprogram(name: "applyIfProperEncoding<float, UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1716 = !DILocation(line: 136, column: 9, scope: !1715)
!1717 = !DILocation(line: 136, column: 34, scope: !1715)
!1718 = !DILocation(line: 136, column: 17, scope: !1715)
!1719 = !DILocation(line: 136, column: 14, scope: !1715)
!1720 = !DILocation(line: 137, column: 20, scope: !1715)
!1721 = !DILocation(line: 138, column: 9, scope: !1715)
!1722 = !DILocation(line: 140, column: 5, scope: !1715)
!1723 = !DILocation(line: 141, column: 1, scope: !1715)
!1724 = distinct !DISubprogram(name: "applyIfProperEncoding<long long, UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1725 = !DILocation(line: 136, column: 9, scope: !1724)
!1726 = !DILocation(line: 136, column: 34, scope: !1724)
!1727 = !DILocation(line: 136, column: 17, scope: !1724)
!1728 = !DILocation(line: 136, column: 14, scope: !1724)
!1729 = !DILocation(line: 137, column: 20, scope: !1724)
!1730 = !DILocation(line: 138, column: 9, scope: !1724)
!1731 = !DILocation(line: 140, column: 5, scope: !1724)
!1732 = !DILocation(line: 141, column: 1, scope: !1724)
!1733 = distinct !DISubprogram(name: "applyIfProperEncoding<unsigned long long, UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1734 = !DILocation(line: 136, column: 9, scope: !1733)
!1735 = !DILocation(line: 136, column: 34, scope: !1733)
!1736 = !DILocation(line: 136, column: 17, scope: !1733)
!1737 = !DILocation(line: 136, column: 14, scope: !1733)
!1738 = !DILocation(line: 137, column: 20, scope: !1733)
!1739 = !DILocation(line: 138, column: 9, scope: !1733)
!1740 = !DILocation(line: 140, column: 5, scope: !1733)
!1741 = !DILocation(line: 141, column: 1, scope: !1733)
!1742 = distinct !DISubprogram(name: "applyIfProperEncoding<double, UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1743 = !DILocation(line: 136, column: 9, scope: !1742)
!1744 = !DILocation(line: 136, column: 34, scope: !1742)
!1745 = !DILocation(line: 136, column: 17, scope: !1742)
!1746 = !DILocation(line: 136, column: 14, scope: !1742)
!1747 = !DILocation(line: 137, column: 20, scope: !1742)
!1748 = !DILocation(line: 138, column: 9, scope: !1742)
!1749 = !DILocation(line: 140, column: 5, scope: !1742)
!1750 = !DILocation(line: 141, column: 1, scope: !1742)
!1751 = distinct !DISubprogram(name: "applyIfProperEncoding<void *, UniformWriter, PrintInfo::Encoding4Uniform>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1752 = !DILocation(line: 136, column: 9, scope: !1751)
!1753 = !DILocation(line: 136, column: 34, scope: !1751)
!1754 = !DILocation(line: 136, column: 17, scope: !1751)
!1755 = !DILocation(line: 136, column: 14, scope: !1751)
!1756 = !DILocation(line: 137, column: 20, scope: !1751)
!1757 = !DILocation(line: 138, column: 9, scope: !1751)
!1758 = !DILocation(line: 140, column: 5, scope: !1751)
!1759 = !DILocation(line: 141, column: 1, scope: !1751)
!1760 = distinct !DISubprogram(name: "call<bool>", scope: !192, file: !192, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1761 = !DILocation(line: 103, column: 68, scope: !1760)
!1762 = !DILocation(line: 103, column: 61, scope: !1760)
!1763 = distinct !DISubprogram(name: "call<bool>", scope: !1, file: !1, line: 352, type: !16, scopeLine: 352, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1764 = !DILocation(line: 352, column: 51, scope: !1763)
!1765 = !DILocation(line: 352, column: 66, scope: !1763)
!1766 = distinct !DISubprogram(name: "WriteArg<bool>", scope: !1, file: !1, line: 372, type: !16, scopeLine: 372, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1767 = !DILocation(line: 374, column: 13, scope: !1766)
!1768 = !DILocation(line: 375, column: 19, scope: !1766)
!1769 = !DILocation(line: 375, column: 17, scope: !1766)
!1770 = !DILocation(line: 375, column: 13, scope: !1766)
!1771 = !DILocation(line: 377, column: 19, scope: !1766)
!1772 = !DILocation(line: 377, column: 17, scope: !1766)
!1773 = !DILocation(line: 378, column: 41, scope: !1766)
!1774 = !{!212, !18, i64 0}
!1775 = !DILocation(line: 378, column: 49, scope: !1766)
!1776 = !DILocation(line: 378, column: 9, scope: !1766)
!1777 = !DILocation(line: 379, column: 5, scope: !1766)
!1778 = distinct !DISubprogram(name: "GetElementaryArg", scope: !1, file: !1, line: 355, type: !16, scopeLine: 355, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1779 = !DILocation(line: 355, column: 49, scope: !1778)
!1780 = !{!212, !18, i64 4}
!1781 = !DILocation(line: 355, column: 53, scope: !1778)
!1782 = !DILocation(line: 355, column: 48, scope: !1778)
!1783 = !DILocation(line: 355, column: 41, scope: !1778)
!1784 = distinct !DISubprogram(name: "write_arg_with_promotion<char const (&)[1]>", scope: !1, file: !1, line: 316, type: !16, scopeLine: 316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1785 = !DILocation(line: 317, column: 32, scope: !1784)
!1786 = !DILocation(line: 317, column: 21, scope: !1784)
!1787 = !DILocation(line: 318, column: 38, scope: !1784)
!1788 = !DILocation(line: 318, column: 26, scope: !1784)
!1789 = !DILocation(line: 318, column: 47, scope: !1784)
!1790 = !DILocation(line: 318, column: 5, scope: !1784)
!1791 = !DILocation(line: 319, column: 5, scope: !1784)
!1792 = !DILocation(line: 319, column: 12, scope: !1784)
!1793 = !DILocation(line: 321, column: 32, scope: !1784)
!1794 = !DILocation(line: 321, column: 25, scope: !1784)
!1795 = !DILocation(line: 322, column: 42, scope: !1784)
!1796 = !DILocation(line: 322, column: 30, scope: !1784)
!1797 = !DILocation(line: 322, column: 51, scope: !1784)
!1798 = !DILocation(line: 322, column: 9, scope: !1784)
!1799 = !DILocation(line: 323, column: 9, scope: !1784)
!1800 = !DILocation(line: 323, column: 16, scope: !1784)
!1801 = !DILocation(line: 335, column: 1, scope: !1784)
!1802 = distinct !DISubprogram(name: "call<int>", scope: !192, file: !192, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1803 = !DILocation(line: 103, column: 68, scope: !1802)
!1804 = !DILocation(line: 103, column: 61, scope: !1802)
!1805 = distinct !DISubprogram(name: "call<int>", scope: !1, file: !1, line: 352, type: !16, scopeLine: 352, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1806 = !DILocation(line: 352, column: 51, scope: !1805)
!1807 = !DILocation(line: 352, column: 66, scope: !1805)
!1808 = distinct !DISubprogram(name: "WriteArg<int>", scope: !1, file: !1, line: 359, type: !16, scopeLine: 359, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1809 = !DILocation(line: 366, column: 19, scope: !1808)
!1810 = !DILocation(line: 366, column: 17, scope: !1808)
!1811 = !DILocation(line: 367, column: 18, scope: !1808)
!1812 = !DILocation(line: 369, column: 37, scope: !1808)
!1813 = !DILocation(line: 369, column: 45, scope: !1808)
!1814 = !DILocation(line: 369, column: 50, scope: !1808)
!1815 = !DILocation(line: 369, column: 9, scope: !1808)
!1816 = !DILocation(line: 370, column: 5, scope: !1808)
!1817 = distinct !DISubprogram(name: "write_arg_with_promotion<int>", scope: !1, file: !1, line: 316, type: !16, scopeLine: 316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1818 = !DILocation(line: 317, column: 32, scope: !1817)
!1819 = !DILocation(line: 317, column: 21, scope: !1817)
!1820 = !DILocation(line: 318, column: 38, scope: !1817)
!1821 = !DILocation(line: 318, column: 26, scope: !1817)
!1822 = !DILocation(line: 318, column: 47, scope: !1817)
!1823 = !DILocation(line: 318, column: 5, scope: !1817)
!1824 = !DILocation(line: 319, column: 5, scope: !1817)
!1825 = !DILocation(line: 319, column: 12, scope: !1817)
!1826 = !DILocation(line: 321, column: 32, scope: !1817)
!1827 = !DILocation(line: 321, column: 25, scope: !1817)
!1828 = !DILocation(line: 322, column: 42, scope: !1817)
!1829 = !DILocation(line: 322, column: 30, scope: !1817)
!1830 = !DILocation(line: 322, column: 51, scope: !1817)
!1831 = !DILocation(line: 322, column: 9, scope: !1817)
!1832 = !DILocation(line: 323, column: 9, scope: !1817)
!1833 = !DILocation(line: 323, column: 16, scope: !1817)
!1834 = !DILocation(line: 335, column: 1, scope: !1817)
!1835 = distinct !DISubprogram(name: "call<unsigned int>", scope: !192, file: !192, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1836 = !DILocation(line: 103, column: 68, scope: !1835)
!1837 = !DILocation(line: 103, column: 61, scope: !1835)
!1838 = distinct !DISubprogram(name: "call<unsigned int>", scope: !1, file: !1, line: 352, type: !16, scopeLine: 352, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1839 = !DILocation(line: 352, column: 51, scope: !1838)
!1840 = !DILocation(line: 352, column: 66, scope: !1838)
!1841 = distinct !DISubprogram(name: "WriteArg<unsigned int>", scope: !1, file: !1, line: 359, type: !16, scopeLine: 359, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1842 = !DILocation(line: 366, column: 19, scope: !1841)
!1843 = !DILocation(line: 366, column: 17, scope: !1841)
!1844 = !DILocation(line: 367, column: 18, scope: !1841)
!1845 = !DILocation(line: 369, column: 37, scope: !1841)
!1846 = !DILocation(line: 369, column: 45, scope: !1841)
!1847 = !DILocation(line: 369, column: 50, scope: !1841)
!1848 = !DILocation(line: 369, column: 9, scope: !1841)
!1849 = !DILocation(line: 370, column: 5, scope: !1841)
!1850 = distinct !DISubprogram(name: "write_arg_with_promotion<unsigned int>", scope: !1, file: !1, line: 316, type: !16, scopeLine: 316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1851 = !DILocation(line: 317, column: 32, scope: !1850)
!1852 = !DILocation(line: 317, column: 21, scope: !1850)
!1853 = !DILocation(line: 318, column: 38, scope: !1850)
!1854 = !DILocation(line: 318, column: 26, scope: !1850)
!1855 = !DILocation(line: 318, column: 47, scope: !1850)
!1856 = !DILocation(line: 318, column: 5, scope: !1850)
!1857 = !DILocation(line: 319, column: 5, scope: !1850)
!1858 = !DILocation(line: 319, column: 12, scope: !1850)
!1859 = !DILocation(line: 321, column: 32, scope: !1850)
!1860 = !DILocation(line: 321, column: 25, scope: !1850)
!1861 = !DILocation(line: 322, column: 42, scope: !1850)
!1862 = !DILocation(line: 322, column: 30, scope: !1850)
!1863 = !DILocation(line: 322, column: 51, scope: !1850)
!1864 = !DILocation(line: 322, column: 9, scope: !1850)
!1865 = !DILocation(line: 323, column: 9, scope: !1850)
!1866 = !DILocation(line: 323, column: 16, scope: !1850)
!1867 = !DILocation(line: 335, column: 1, scope: !1850)
!1868 = distinct !DISubprogram(name: "call<float>", scope: !192, file: !192, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1869 = !DILocation(line: 103, column: 68, scope: !1868)
!1870 = !DILocation(line: 103, column: 61, scope: !1868)
!1871 = distinct !DISubprogram(name: "call<float>", scope: !1, file: !1, line: 352, type: !16, scopeLine: 352, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1872 = !DILocation(line: 352, column: 51, scope: !1871)
!1873 = !DILocation(line: 352, column: 66, scope: !1871)
!1874 = distinct !DISubprogram(name: "WriteArg<float>", scope: !1, file: !1, line: 359, type: !16, scopeLine: 359, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1875 = !DILocation(line: 366, column: 19, scope: !1874)
!1876 = !DILocation(line: 366, column: 17, scope: !1874)
!1877 = !DILocation(line: 367, column: 18, scope: !1874)
!1878 = !DILocation(line: 369, column: 37, scope: !1874)
!1879 = !DILocation(line: 369, column: 45, scope: !1874)
!1880 = !DILocation(line: 369, column: 50, scope: !1874)
!1881 = !DILocation(line: 369, column: 9, scope: !1874)
!1882 = !DILocation(line: 370, column: 5, scope: !1874)
!1883 = distinct !DISubprogram(name: "write_arg_with_promotion<float>", scope: !1, file: !1, line: 316, type: !16, scopeLine: 316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1884 = !DILocation(line: 317, column: 32, scope: !1883)
!1885 = !DILocation(line: 317, column: 21, scope: !1883)
!1886 = !DILocation(line: 318, column: 38, scope: !1883)
!1887 = !DILocation(line: 318, column: 26, scope: !1883)
!1888 = !DILocation(line: 318, column: 47, scope: !1883)
!1889 = !DILocation(line: 318, column: 5, scope: !1883)
!1890 = !DILocation(line: 319, column: 5, scope: !1883)
!1891 = !DILocation(line: 319, column: 12, scope: !1883)
!1892 = !DILocation(line: 321, column: 32, scope: !1883)
!1893 = !DILocation(line: 321, column: 25, scope: !1883)
!1894 = !DILocation(line: 322, column: 42, scope: !1883)
!1895 = !DILocation(line: 322, column: 30, scope: !1883)
!1896 = !DILocation(line: 322, column: 51, scope: !1883)
!1897 = !DILocation(line: 322, column: 9, scope: !1883)
!1898 = !DILocation(line: 323, column: 9, scope: !1883)
!1899 = !DILocation(line: 323, column: 16, scope: !1883)
!1900 = !DILocation(line: 335, column: 1, scope: !1883)
!1901 = distinct !DISubprogram(name: "call<long long>", scope: !192, file: !192, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1902 = !DILocation(line: 103, column: 68, scope: !1901)
!1903 = !DILocation(line: 103, column: 61, scope: !1901)
!1904 = distinct !DISubprogram(name: "call<long long>", scope: !1, file: !1, line: 352, type: !16, scopeLine: 352, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1905 = !DILocation(line: 352, column: 51, scope: !1904)
!1906 = !DILocation(line: 352, column: 66, scope: !1904)
!1907 = distinct !DISubprogram(name: "WriteArg<long long>", scope: !1, file: !1, line: 359, type: !16, scopeLine: 359, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1908 = !DILocation(line: 363, column: 19, scope: !1907)
!1909 = !DILocation(line: 363, column: 17, scope: !1907)
!1910 = !DILocation(line: 364, column: 20, scope: !1907)
!1911 = !DILocation(line: 364, column: 18, scope: !1907)
!1912 = !DILocation(line: 369, column: 37, scope: !1907)
!1913 = !DILocation(line: 369, column: 45, scope: !1907)
!1914 = !DILocation(line: 369, column: 50, scope: !1907)
!1915 = !DILocation(line: 369, column: 9, scope: !1907)
!1916 = !DILocation(line: 370, column: 5, scope: !1907)
!1917 = distinct !DISubprogram(name: "write_arg_with_promotion<long long>", scope: !1, file: !1, line: 316, type: !16, scopeLine: 316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1918 = !DILocation(line: 317, column: 32, scope: !1917)
!1919 = !DILocation(line: 317, column: 21, scope: !1917)
!1920 = !DILocation(line: 318, column: 38, scope: !1917)
!1921 = !DILocation(line: 318, column: 26, scope: !1917)
!1922 = !DILocation(line: 318, column: 47, scope: !1917)
!1923 = !DILocation(line: 318, column: 5, scope: !1917)
!1924 = !DILocation(line: 319, column: 5, scope: !1917)
!1925 = !DILocation(line: 319, column: 12, scope: !1917)
!1926 = !DILocation(line: 327, column: 19, scope: !1917)
!1927 = !DILocation(line: 327, column: 17, scope: !1917)
!1928 = !DILocation(line: 328, column: 19, scope: !1917)
!1929 = !DILocation(line: 328, column: 17, scope: !1917)
!1930 = !DILocation(line: 330, column: 19, scope: !1917)
!1931 = !DILocation(line: 330, column: 17, scope: !1917)
!1932 = !DILocation(line: 331, column: 19, scope: !1917)
!1933 = !DILocation(line: 331, column: 26, scope: !1917)
!1934 = !DILocation(line: 331, column: 17, scope: !1917)
!1935 = !DILocation(line: 332, column: 30, scope: !1917)
!1936 = !DILocation(line: 332, column: 36, scope: !1917)
!1937 = !DILocation(line: 332, column: 9, scope: !1917)
!1938 = !DILocation(line: 333, column: 9, scope: !1917)
!1939 = !DILocation(line: 333, column: 16, scope: !1917)
!1940 = !DILocation(line: 335, column: 1, scope: !1917)
!1941 = distinct !DISubprogram(name: "call<unsigned long long>", scope: !192, file: !192, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1942 = !DILocation(line: 103, column: 68, scope: !1941)
!1943 = !DILocation(line: 103, column: 61, scope: !1941)
!1944 = distinct !DISubprogram(name: "call<unsigned long long>", scope: !1, file: !1, line: 352, type: !16, scopeLine: 352, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1945 = !DILocation(line: 352, column: 51, scope: !1944)
!1946 = !DILocation(line: 352, column: 66, scope: !1944)
!1947 = distinct !DISubprogram(name: "WriteArg<unsigned long long>", scope: !1, file: !1, line: 359, type: !16, scopeLine: 359, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1948 = !DILocation(line: 363, column: 19, scope: !1947)
!1949 = !DILocation(line: 363, column: 17, scope: !1947)
!1950 = !DILocation(line: 364, column: 20, scope: !1947)
!1951 = !DILocation(line: 364, column: 18, scope: !1947)
!1952 = !DILocation(line: 369, column: 37, scope: !1947)
!1953 = !DILocation(line: 369, column: 45, scope: !1947)
!1954 = !DILocation(line: 369, column: 50, scope: !1947)
!1955 = !DILocation(line: 369, column: 9, scope: !1947)
!1956 = !DILocation(line: 370, column: 5, scope: !1947)
!1957 = distinct !DISubprogram(name: "write_arg_with_promotion<unsigned long long>", scope: !1, file: !1, line: 316, type: !16, scopeLine: 316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1958 = !DILocation(line: 317, column: 32, scope: !1957)
!1959 = !DILocation(line: 317, column: 21, scope: !1957)
!1960 = !DILocation(line: 318, column: 38, scope: !1957)
!1961 = !DILocation(line: 318, column: 26, scope: !1957)
!1962 = !DILocation(line: 318, column: 47, scope: !1957)
!1963 = !DILocation(line: 318, column: 5, scope: !1957)
!1964 = !DILocation(line: 319, column: 5, scope: !1957)
!1965 = !DILocation(line: 319, column: 12, scope: !1957)
!1966 = !DILocation(line: 327, column: 19, scope: !1957)
!1967 = !DILocation(line: 327, column: 17, scope: !1957)
!1968 = !DILocation(line: 328, column: 19, scope: !1957)
!1969 = !DILocation(line: 328, column: 17, scope: !1957)
!1970 = !DILocation(line: 330, column: 19, scope: !1957)
!1971 = !DILocation(line: 330, column: 17, scope: !1957)
!1972 = !DILocation(line: 331, column: 19, scope: !1957)
!1973 = !DILocation(line: 331, column: 26, scope: !1957)
!1974 = !DILocation(line: 331, column: 17, scope: !1957)
!1975 = !DILocation(line: 332, column: 30, scope: !1957)
!1976 = !DILocation(line: 332, column: 36, scope: !1957)
!1977 = !DILocation(line: 332, column: 9, scope: !1957)
!1978 = !DILocation(line: 333, column: 9, scope: !1957)
!1979 = !DILocation(line: 333, column: 16, scope: !1957)
!1980 = !DILocation(line: 335, column: 1, scope: !1957)
!1981 = distinct !DISubprogram(name: "call<double>", scope: !192, file: !192, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1982 = !DILocation(line: 103, column: 68, scope: !1981)
!1983 = !DILocation(line: 103, column: 61, scope: !1981)
!1984 = distinct !DISubprogram(name: "call<double>", scope: !1, file: !1, line: 352, type: !16, scopeLine: 352, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1985 = !DILocation(line: 352, column: 51, scope: !1984)
!1986 = !DILocation(line: 352, column: 66, scope: !1984)
!1987 = distinct !DISubprogram(name: "WriteArg<double>", scope: !1, file: !1, line: 359, type: !16, scopeLine: 359, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1988 = !DILocation(line: 363, column: 19, scope: !1987)
!1989 = !DILocation(line: 363, column: 17, scope: !1987)
!1990 = !DILocation(line: 364, column: 20, scope: !1987)
!1991 = !DILocation(line: 364, column: 18, scope: !1987)
!1992 = !DILocation(line: 369, column: 37, scope: !1987)
!1993 = !DILocation(line: 369, column: 45, scope: !1987)
!1994 = !DILocation(line: 369, column: 50, scope: !1987)
!1995 = !DILocation(line: 369, column: 9, scope: !1987)
!1996 = !DILocation(line: 370, column: 5, scope: !1987)
!1997 = distinct !DISubprogram(name: "write_arg_with_promotion<double>", scope: !1, file: !1, line: 316, type: !16, scopeLine: 316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!1998 = !DILocation(line: 317, column: 32, scope: !1997)
!1999 = !DILocation(line: 317, column: 21, scope: !1997)
!2000 = !DILocation(line: 318, column: 38, scope: !1997)
!2001 = !DILocation(line: 318, column: 26, scope: !1997)
!2002 = !DILocation(line: 318, column: 47, scope: !1997)
!2003 = !DILocation(line: 318, column: 5, scope: !1997)
!2004 = !DILocation(line: 319, column: 5, scope: !1997)
!2005 = !DILocation(line: 319, column: 12, scope: !1997)
!2006 = !DILocation(line: 327, column: 19, scope: !1997)
!2007 = !DILocation(line: 327, column: 17, scope: !1997)
!2008 = !DILocation(line: 328, column: 19, scope: !1997)
!2009 = !DILocation(line: 328, column: 17, scope: !1997)
!2010 = !DILocation(line: 330, column: 19, scope: !1997)
!2011 = !DILocation(line: 330, column: 17, scope: !1997)
!2012 = !DILocation(line: 331, column: 19, scope: !1997)
!2013 = !DILocation(line: 331, column: 26, scope: !1997)
!2014 = !DILocation(line: 331, column: 17, scope: !1997)
!2015 = !DILocation(line: 332, column: 30, scope: !1997)
!2016 = !DILocation(line: 332, column: 36, scope: !1997)
!2017 = !DILocation(line: 332, column: 9, scope: !1997)
!2018 = !DILocation(line: 333, column: 9, scope: !1997)
!2019 = !DILocation(line: 333, column: 16, scope: !1997)
!2020 = !DILocation(line: 335, column: 1, scope: !1997)
!2021 = distinct !DISubprogram(name: "call<void *>", scope: !192, file: !192, line: 103, type: !16, scopeLine: 103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2022 = !DILocation(line: 103, column: 68, scope: !2021)
!2023 = !DILocation(line: 103, column: 61, scope: !2021)
!2024 = distinct !DISubprogram(name: "call<void *>", scope: !1, file: !1, line: 352, type: !16, scopeLine: 352, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2025 = !DILocation(line: 352, column: 51, scope: !2024)
!2026 = !DILocation(line: 352, column: 66, scope: !2024)
!2027 = distinct !DISubprogram(name: "WriteArg<void *>", scope: !1, file: !1, line: 359, type: !16, scopeLine: 359, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2028 = !DILocation(line: 363, column: 19, scope: !2027)
!2029 = !DILocation(line: 363, column: 17, scope: !2027)
!2030 = !DILocation(line: 364, column: 20, scope: !2027)
!2031 = !DILocation(line: 364, column: 18, scope: !2027)
!2032 = !DILocation(line: 369, column: 37, scope: !2027)
!2033 = !DILocation(line: 369, column: 45, scope: !2027)
!2034 = !DILocation(line: 369, column: 50, scope: !2027)
!2035 = !DILocation(line: 369, column: 9, scope: !2027)
!2036 = !DILocation(line: 370, column: 5, scope: !2027)
!2037 = distinct !DISubprogram(name: "write_arg_with_promotion<void *>", scope: !1, file: !1, line: 316, type: !16, scopeLine: 316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2038 = !DILocation(line: 317, column: 32, scope: !2037)
!2039 = !DILocation(line: 317, column: 21, scope: !2037)
!2040 = !DILocation(line: 318, column: 38, scope: !2037)
!2041 = !DILocation(line: 318, column: 26, scope: !2037)
!2042 = !DILocation(line: 318, column: 47, scope: !2037)
!2043 = !DILocation(line: 318, column: 5, scope: !2037)
!2044 = !DILocation(line: 319, column: 5, scope: !2037)
!2045 = !DILocation(line: 319, column: 12, scope: !2037)
!2046 = !DILocation(line: 321, column: 32, scope: !2037)
!2047 = !DILocation(line: 321, column: 25, scope: !2037)
!2048 = !DILocation(line: 322, column: 42, scope: !2037)
!2049 = !DILocation(line: 322, column: 30, scope: !2037)
!2050 = !DILocation(line: 322, column: 51, scope: !2037)
!2051 = !DILocation(line: 322, column: 9, scope: !2037)
!2052 = !DILocation(line: 323, column: 9, scope: !2037)
!2053 = !DILocation(line: 323, column: 16, scope: !2037)
!2054 = !DILocation(line: 335, column: 1, scope: !2037)
!2055 = distinct !DISubprogram(name: "switchEncoding<VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 143, type: !16, scopeLine: 143, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2056 = !DILocation(line: 144, column: 40, scope: !2055)
!2057 = !DILocation(line: 144, column: 46, scope: !2055)
!2058 = !DILocation(line: 144, column: 12, scope: !2055)
!2059 = !DILocation(line: 144, column: 58, scope: !2055)
!2060 = !DILocation(line: 144, column: 88, scope: !2055)
!2061 = !DILocation(line: 144, column: 94, scope: !2055)
!2062 = !DILocation(line: 144, column: 61, scope: !2055)
!2063 = !DILocation(line: 144, column: 106, scope: !2055)
!2064 = !DILocation(line: 145, column: 44, scope: !2055)
!2065 = !DILocation(line: 145, column: 50, scope: !2055)
!2066 = !DILocation(line: 145, column: 12, scope: !2055)
!2067 = !DILocation(line: 145, column: 62, scope: !2055)
!2068 = !DILocation(line: 145, column: 94, scope: !2055)
!2069 = !DILocation(line: 145, column: 100, scope: !2055)
!2070 = !DILocation(line: 145, column: 65, scope: !2055)
!2071 = !DILocation(line: 145, column: 112, scope: !2055)
!2072 = !DILocation(line: 146, column: 45, scope: !2055)
!2073 = !DILocation(line: 146, column: 51, scope: !2055)
!2074 = !DILocation(line: 146, column: 12, scope: !2055)
!2075 = !DILocation(line: 146, column: 63, scope: !2055)
!2076 = !DILocation(line: 147, column: 54, scope: !2055)
!2077 = !DILocation(line: 147, column: 60, scope: !2055)
!2078 = !DILocation(line: 147, column: 12, scope: !2055)
!2079 = !DILocation(line: 147, column: 72, scope: !2055)
!2080 = !DILocation(line: 148, column: 42, scope: !2055)
!2081 = !DILocation(line: 148, column: 48, scope: !2055)
!2082 = !DILocation(line: 148, column: 12, scope: !2055)
!2083 = !DILocation(line: 148, column: 60, scope: !2055)
!2084 = !DILocation(line: 148, column: 93, scope: !2055)
!2085 = !DILocation(line: 148, column: 99, scope: !2055)
!2086 = !DILocation(line: 148, column: 63, scope: !2055)
!2087 = !DILocation(line: 144, column: 5, scope: !2055)
!2088 = distinct !DISubprogram(name: "applyIfProperEncoding<bool, VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2089 = !DILocation(line: 136, column: 9, scope: !2088)
!2090 = !DILocation(line: 136, column: 34, scope: !2088)
!2091 = !DILocation(line: 136, column: 17, scope: !2088)
!2092 = !DILocation(line: 136, column: 14, scope: !2088)
!2093 = !DILocation(line: 137, column: 20, scope: !2088)
!2094 = !DILocation(line: 138, column: 9, scope: !2088)
!2095 = !DILocation(line: 140, column: 5, scope: !2088)
!2096 = !DILocation(line: 141, column: 1, scope: !2088)
!2097 = distinct !DISubprogram(name: "applyIfProperEncoding<int, VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2098 = !DILocation(line: 136, column: 9, scope: !2097)
!2099 = !DILocation(line: 136, column: 34, scope: !2097)
!2100 = !DILocation(line: 136, column: 17, scope: !2097)
!2101 = !DILocation(line: 136, column: 14, scope: !2097)
!2102 = !DILocation(line: 137, column: 20, scope: !2097)
!2103 = !DILocation(line: 138, column: 9, scope: !2097)
!2104 = !DILocation(line: 140, column: 5, scope: !2097)
!2105 = !DILocation(line: 141, column: 1, scope: !2097)
!2106 = distinct !DISubprogram(name: "applyIfProperEncoding<unsigned int, VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2107 = !DILocation(line: 136, column: 9, scope: !2106)
!2108 = !DILocation(line: 136, column: 34, scope: !2106)
!2109 = !DILocation(line: 136, column: 17, scope: !2106)
!2110 = !DILocation(line: 136, column: 14, scope: !2106)
!2111 = !DILocation(line: 137, column: 20, scope: !2106)
!2112 = !DILocation(line: 138, column: 9, scope: !2106)
!2113 = !DILocation(line: 140, column: 5, scope: !2106)
!2114 = !DILocation(line: 141, column: 1, scope: !2106)
!2115 = distinct !DISubprogram(name: "applyIfProperEncoding<float, VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2116 = !DILocation(line: 136, column: 9, scope: !2115)
!2117 = !DILocation(line: 136, column: 34, scope: !2115)
!2118 = !DILocation(line: 136, column: 17, scope: !2115)
!2119 = !DILocation(line: 136, column: 14, scope: !2115)
!2120 = !DILocation(line: 137, column: 20, scope: !2115)
!2121 = !DILocation(line: 138, column: 9, scope: !2115)
!2122 = !DILocation(line: 140, column: 5, scope: !2115)
!2123 = !DILocation(line: 141, column: 1, scope: !2115)
!2124 = distinct !DISubprogram(name: "applyIfProperEncoding<long long, VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2125 = !DILocation(line: 136, column: 9, scope: !2124)
!2126 = !DILocation(line: 136, column: 34, scope: !2124)
!2127 = !DILocation(line: 136, column: 17, scope: !2124)
!2128 = !DILocation(line: 136, column: 14, scope: !2124)
!2129 = !DILocation(line: 137, column: 20, scope: !2124)
!2130 = !DILocation(line: 138, column: 9, scope: !2124)
!2131 = !DILocation(line: 140, column: 5, scope: !2124)
!2132 = !DILocation(line: 141, column: 1, scope: !2124)
!2133 = distinct !DISubprogram(name: "applyIfProperEncoding<unsigned long long, VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2134 = !DILocation(line: 136, column: 9, scope: !2133)
!2135 = !DILocation(line: 136, column: 34, scope: !2133)
!2136 = !DILocation(line: 136, column: 17, scope: !2133)
!2137 = !DILocation(line: 136, column: 14, scope: !2133)
!2138 = !DILocation(line: 137, column: 20, scope: !2133)
!2139 = !DILocation(line: 138, column: 9, scope: !2133)
!2140 = !DILocation(line: 140, column: 5, scope: !2133)
!2141 = !DILocation(line: 141, column: 1, scope: !2133)
!2142 = distinct !DISubprogram(name: "applyIfProperEncoding<double, VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2143 = !DILocation(line: 136, column: 9, scope: !2142)
!2144 = !DILocation(line: 136, column: 34, scope: !2142)
!2145 = !DILocation(line: 136, column: 17, scope: !2142)
!2146 = !DILocation(line: 136, column: 14, scope: !2142)
!2147 = !DILocation(line: 137, column: 20, scope: !2142)
!2148 = !DILocation(line: 138, column: 9, scope: !2142)
!2149 = !DILocation(line: 140, column: 5, scope: !2142)
!2150 = !DILocation(line: 141, column: 1, scope: !2142)
!2151 = distinct !DISubprogram(name: "applyIfProperEncoding<void *, VaryingWriter, PrintInfo::Encoding4Varying>", scope: !192, file: !192, line: 135, type: !16, scopeLine: 135, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2152 = !DILocation(line: 136, column: 9, scope: !2151)
!2153 = !DILocation(line: 136, column: 34, scope: !2151)
!2154 = !DILocation(line: 136, column: 17, scope: !2151)
!2155 = !DILocation(line: 136, column: 14, scope: !2151)
!2156 = !DILocation(line: 137, column: 20, scope: !2151)
!2157 = !DILocation(line: 138, column: 9, scope: !2151)
!2158 = !DILocation(line: 140, column: 5, scope: !2151)
!2159 = !DILocation(line: 141, column: 1, scope: !2151)
!2160 = distinct !DISubprogram(name: "call<bool>", scope: !192, file: !192, line: 108, type: !16, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2161 = !DILocation(line: 108, column: 68, scope: !2160)
!2162 = !DILocation(line: 108, column: 61, scope: !2160)
!2163 = distinct !DISubprogram(name: "call<bool>", scope: !1, file: !1, line: 394, type: !16, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2164 = !DILocation(line: 394, column: 51, scope: !2163)
!2165 = !DILocation(line: 394, column: 66, scope: !2163)
!2166 = distinct !DISubprogram(name: "WriteArg<bool>", scope: !1, file: !1, line: 397, type: !16, scopeLine: 397, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2167 = !DILocation(line: 398, column: 18, scope: !2166)
!2168 = !DILocation(line: 398, column: 14, scope: !2166)
!2169 = !DILocation(line: 398, column: 28, scope: !2166)
!2170 = !DILocation(line: 398, column: 35, scope: !2166)
!2171 = !DILocation(line: 398, column: 33, scope: !2166)
!2172 = !DILocation(line: 398, column: 9, scope: !2166)
!2173 = !DILocation(line: 401, column: 17, scope: !2166)
!2174 = !DILocation(line: 401, column: 33, scope: !2166)
!2175 = !DILocation(line: 401, column: 30, scope: !2166)
!2176 = !DILocation(line: 401, column: 22, scope: !2166)
!2177 = !DILocation(line: 402, column: 40, scope: !2166)
!2178 = !DILocation(line: 402, column: 38, scope: !2166)
!2179 = !DILocation(line: 403, column: 41, scope: !2166)
!2180 = !DILocation(line: 403, column: 39, scope: !2166)
!2181 = !DILocation(line: 404, column: 13, scope: !2166)
!2182 = !DILocation(line: 405, column: 40, scope: !2166)
!2183 = !DILocation(line: 405, column: 38, scope: !2166)
!2184 = !DILocation(line: 406, column: 41, scope: !2166)
!2185 = !DILocation(line: 406, column: 39, scope: !2166)
!2186 = !DILocation(line: 408, column: 45, scope: !2166)
!2187 = !{!228, !18, i64 0}
!2188 = !DILocation(line: 408, column: 53, scope: !2166)
!2189 = !DILocation(line: 408, column: 13, scope: !2166)
!2190 = !DILocation(line: 409, column: 13, scope: !2166)
!2191 = !DILocation(line: 410, column: 45, scope: !2166)
!2192 = !DILocation(line: 410, column: 53, scope: !2166)
!2193 = !DILocation(line: 410, column: 13, scope: !2166)
!2194 = !DILocation(line: 411, column: 9, scope: !2166)
!2195 = !DILocation(line: 398, column: 42, scope: !2166)
!2196 = distinct !{!2196, !2172, !2194}
!2197 = !DILocation(line: 412, column: 5, scope: !2166)
!2198 = distinct !DISubprogram(name: "WriteVecElem<bool>", scope: !1, file: !1, line: 414, type: !16, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2199 = !DILocation(line: 414, column: 73, scope: !2198)
!2200 = !DILocation(line: 414, column: 81, scope: !2198)
!2201 = !{!228, !18, i64 4}
!2202 = !DILocation(line: 414, column: 87, scope: !2198)
!2203 = !DILocation(line: 414, column: 94, scope: !2198)
!2204 = !DILocation(line: 414, column: 100, scope: !2198)
!2205 = !DILocation(line: 414, column: 59, scope: !2198)
!2206 = !DILocation(line: 414, column: 108, scope: !2198)
!2207 = !DILocation(line: 414, column: 119, scope: !2198)
!2208 = distinct !DISubprogram(name: "call<int>", scope: !192, file: !192, line: 108, type: !16, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2209 = !DILocation(line: 108, column: 68, scope: !2208)
!2210 = !DILocation(line: 108, column: 61, scope: !2208)
!2211 = distinct !DISubprogram(name: "call<int>", scope: !1, file: !1, line: 394, type: !16, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2212 = !DILocation(line: 394, column: 51, scope: !2211)
!2213 = !DILocation(line: 394, column: 66, scope: !2211)
!2214 = distinct !DISubprogram(name: "WriteArg<int>", scope: !1, file: !1, line: 397, type: !16, scopeLine: 397, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2215 = !DILocation(line: 398, column: 18, scope: !2214)
!2216 = !DILocation(line: 398, column: 14, scope: !2214)
!2217 = !DILocation(line: 398, column: 28, scope: !2214)
!2218 = !DILocation(line: 398, column: 35, scope: !2214)
!2219 = !DILocation(line: 398, column: 33, scope: !2214)
!2220 = !DILocation(line: 398, column: 9, scope: !2214)
!2221 = !DILocation(line: 401, column: 17, scope: !2214)
!2222 = !DILocation(line: 401, column: 33, scope: !2214)
!2223 = !DILocation(line: 401, column: 30, scope: !2214)
!2224 = !DILocation(line: 401, column: 22, scope: !2214)
!2225 = !DILocation(line: 402, column: 40, scope: !2214)
!2226 = !DILocation(line: 402, column: 38, scope: !2214)
!2227 = !DILocation(line: 403, column: 41, scope: !2214)
!2228 = !DILocation(line: 403, column: 39, scope: !2214)
!2229 = !DILocation(line: 404, column: 13, scope: !2214)
!2230 = !DILocation(line: 405, column: 40, scope: !2214)
!2231 = !DILocation(line: 405, column: 38, scope: !2214)
!2232 = !DILocation(line: 406, column: 41, scope: !2214)
!2233 = !DILocation(line: 406, column: 39, scope: !2214)
!2234 = !DILocation(line: 408, column: 45, scope: !2214)
!2235 = !DILocation(line: 408, column: 53, scope: !2214)
!2236 = !DILocation(line: 408, column: 13, scope: !2214)
!2237 = !DILocation(line: 409, column: 13, scope: !2214)
!2238 = !DILocation(line: 410, column: 45, scope: !2214)
!2239 = !DILocation(line: 410, column: 53, scope: !2214)
!2240 = !DILocation(line: 410, column: 13, scope: !2214)
!2241 = !DILocation(line: 411, column: 9, scope: !2214)
!2242 = !DILocation(line: 398, column: 42, scope: !2214)
!2243 = distinct !{!2243, !2220, !2241}
!2244 = !DILocation(line: 412, column: 5, scope: !2214)
!2245 = distinct !DISubprogram(name: "WriteVecElem<int>", scope: !1, file: !1, line: 414, type: !16, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2246 = !DILocation(line: 414, column: 73, scope: !2245)
!2247 = !DILocation(line: 414, column: 81, scope: !2245)
!2248 = !DILocation(line: 414, column: 87, scope: !2245)
!2249 = !DILocation(line: 414, column: 94, scope: !2245)
!2250 = !DILocation(line: 414, column: 100, scope: !2245)
!2251 = !DILocation(line: 414, column: 59, scope: !2245)
!2252 = !DILocation(line: 414, column: 108, scope: !2245)
!2253 = !DILocation(line: 414, column: 119, scope: !2245)
!2254 = distinct !DISubprogram(name: "call<unsigned int>", scope: !192, file: !192, line: 108, type: !16, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2255 = !DILocation(line: 108, column: 68, scope: !2254)
!2256 = !DILocation(line: 108, column: 61, scope: !2254)
!2257 = distinct !DISubprogram(name: "call<unsigned int>", scope: !1, file: !1, line: 394, type: !16, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2258 = !DILocation(line: 394, column: 51, scope: !2257)
!2259 = !DILocation(line: 394, column: 66, scope: !2257)
!2260 = distinct !DISubprogram(name: "WriteArg<unsigned int>", scope: !1, file: !1, line: 397, type: !16, scopeLine: 397, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2261 = !DILocation(line: 398, column: 18, scope: !2260)
!2262 = !DILocation(line: 398, column: 14, scope: !2260)
!2263 = !DILocation(line: 398, column: 28, scope: !2260)
!2264 = !DILocation(line: 398, column: 35, scope: !2260)
!2265 = !DILocation(line: 398, column: 33, scope: !2260)
!2266 = !DILocation(line: 398, column: 9, scope: !2260)
!2267 = !DILocation(line: 401, column: 17, scope: !2260)
!2268 = !DILocation(line: 401, column: 33, scope: !2260)
!2269 = !DILocation(line: 401, column: 30, scope: !2260)
!2270 = !DILocation(line: 401, column: 22, scope: !2260)
!2271 = !DILocation(line: 402, column: 40, scope: !2260)
!2272 = !DILocation(line: 402, column: 38, scope: !2260)
!2273 = !DILocation(line: 403, column: 41, scope: !2260)
!2274 = !DILocation(line: 403, column: 39, scope: !2260)
!2275 = !DILocation(line: 404, column: 13, scope: !2260)
!2276 = !DILocation(line: 405, column: 40, scope: !2260)
!2277 = !DILocation(line: 405, column: 38, scope: !2260)
!2278 = !DILocation(line: 406, column: 41, scope: !2260)
!2279 = !DILocation(line: 406, column: 39, scope: !2260)
!2280 = !DILocation(line: 408, column: 45, scope: !2260)
!2281 = !DILocation(line: 408, column: 53, scope: !2260)
!2282 = !DILocation(line: 408, column: 13, scope: !2260)
!2283 = !DILocation(line: 409, column: 13, scope: !2260)
!2284 = !DILocation(line: 410, column: 45, scope: !2260)
!2285 = !DILocation(line: 410, column: 53, scope: !2260)
!2286 = !DILocation(line: 410, column: 13, scope: !2260)
!2287 = !DILocation(line: 411, column: 9, scope: !2260)
!2288 = !DILocation(line: 398, column: 42, scope: !2260)
!2289 = distinct !{!2289, !2266, !2287}
!2290 = !DILocation(line: 412, column: 5, scope: !2260)
!2291 = distinct !DISubprogram(name: "WriteVecElem<unsigned int>", scope: !1, file: !1, line: 414, type: !16, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2292 = !DILocation(line: 414, column: 73, scope: !2291)
!2293 = !DILocation(line: 414, column: 81, scope: !2291)
!2294 = !DILocation(line: 414, column: 87, scope: !2291)
!2295 = !DILocation(line: 414, column: 94, scope: !2291)
!2296 = !DILocation(line: 414, column: 100, scope: !2291)
!2297 = !DILocation(line: 414, column: 59, scope: !2291)
!2298 = !DILocation(line: 414, column: 108, scope: !2291)
!2299 = !DILocation(line: 414, column: 119, scope: !2291)
!2300 = distinct !DISubprogram(name: "call<float>", scope: !192, file: !192, line: 108, type: !16, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2301 = !DILocation(line: 108, column: 68, scope: !2300)
!2302 = !DILocation(line: 108, column: 61, scope: !2300)
!2303 = distinct !DISubprogram(name: "call<float>", scope: !1, file: !1, line: 394, type: !16, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2304 = !DILocation(line: 394, column: 51, scope: !2303)
!2305 = !DILocation(line: 394, column: 66, scope: !2303)
!2306 = distinct !DISubprogram(name: "WriteArg<float>", scope: !1, file: !1, line: 397, type: !16, scopeLine: 397, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2307 = !DILocation(line: 398, column: 18, scope: !2306)
!2308 = !DILocation(line: 398, column: 14, scope: !2306)
!2309 = !DILocation(line: 398, column: 28, scope: !2306)
!2310 = !DILocation(line: 398, column: 35, scope: !2306)
!2311 = !DILocation(line: 398, column: 33, scope: !2306)
!2312 = !DILocation(line: 398, column: 9, scope: !2306)
!2313 = !DILocation(line: 401, column: 17, scope: !2306)
!2314 = !DILocation(line: 401, column: 33, scope: !2306)
!2315 = !DILocation(line: 401, column: 30, scope: !2306)
!2316 = !DILocation(line: 401, column: 22, scope: !2306)
!2317 = !DILocation(line: 402, column: 40, scope: !2306)
!2318 = !DILocation(line: 402, column: 38, scope: !2306)
!2319 = !DILocation(line: 403, column: 41, scope: !2306)
!2320 = !DILocation(line: 403, column: 39, scope: !2306)
!2321 = !DILocation(line: 404, column: 13, scope: !2306)
!2322 = !DILocation(line: 405, column: 40, scope: !2306)
!2323 = !DILocation(line: 405, column: 38, scope: !2306)
!2324 = !DILocation(line: 406, column: 41, scope: !2306)
!2325 = !DILocation(line: 406, column: 39, scope: !2306)
!2326 = !DILocation(line: 408, column: 45, scope: !2306)
!2327 = !DILocation(line: 408, column: 53, scope: !2306)
!2328 = !DILocation(line: 408, column: 13, scope: !2306)
!2329 = !DILocation(line: 409, column: 13, scope: !2306)
!2330 = !DILocation(line: 410, column: 45, scope: !2306)
!2331 = !DILocation(line: 410, column: 53, scope: !2306)
!2332 = !DILocation(line: 410, column: 13, scope: !2306)
!2333 = !DILocation(line: 411, column: 9, scope: !2306)
!2334 = !DILocation(line: 398, column: 42, scope: !2306)
!2335 = distinct !{!2335, !2312, !2333}
!2336 = !DILocation(line: 412, column: 5, scope: !2306)
!2337 = distinct !DISubprogram(name: "WriteVecElem<float>", scope: !1, file: !1, line: 414, type: !16, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2338 = !DILocation(line: 414, column: 73, scope: !2337)
!2339 = !DILocation(line: 414, column: 81, scope: !2337)
!2340 = !DILocation(line: 414, column: 87, scope: !2337)
!2341 = !DILocation(line: 414, column: 94, scope: !2337)
!2342 = !DILocation(line: 414, column: 100, scope: !2337)
!2343 = !DILocation(line: 414, column: 59, scope: !2337)
!2344 = !DILocation(line: 414, column: 108, scope: !2337)
!2345 = !DILocation(line: 414, column: 119, scope: !2337)
!2346 = distinct !DISubprogram(name: "call<long long>", scope: !192, file: !192, line: 108, type: !16, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2347 = !DILocation(line: 108, column: 68, scope: !2346)
!2348 = !DILocation(line: 108, column: 61, scope: !2346)
!2349 = distinct !DISubprogram(name: "call<long long>", scope: !1, file: !1, line: 394, type: !16, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2350 = !DILocation(line: 394, column: 51, scope: !2349)
!2351 = !DILocation(line: 394, column: 66, scope: !2349)
!2352 = distinct !DISubprogram(name: "WriteArg<long long>", scope: !1, file: !1, line: 397, type: !16, scopeLine: 397, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2353 = !DILocation(line: 398, column: 18, scope: !2352)
!2354 = !DILocation(line: 398, column: 14, scope: !2352)
!2355 = !DILocation(line: 398, column: 28, scope: !2352)
!2356 = !DILocation(line: 398, column: 35, scope: !2352)
!2357 = !DILocation(line: 398, column: 33, scope: !2352)
!2358 = !DILocation(line: 398, column: 9, scope: !2352)
!2359 = !DILocation(line: 401, column: 17, scope: !2352)
!2360 = !DILocation(line: 401, column: 33, scope: !2352)
!2361 = !DILocation(line: 401, column: 30, scope: !2352)
!2362 = !DILocation(line: 401, column: 22, scope: !2352)
!2363 = !DILocation(line: 402, column: 40, scope: !2352)
!2364 = !DILocation(line: 402, column: 38, scope: !2352)
!2365 = !DILocation(line: 403, column: 41, scope: !2352)
!2366 = !DILocation(line: 403, column: 39, scope: !2352)
!2367 = !DILocation(line: 404, column: 13, scope: !2352)
!2368 = !DILocation(line: 405, column: 40, scope: !2352)
!2369 = !DILocation(line: 405, column: 38, scope: !2352)
!2370 = !DILocation(line: 406, column: 41, scope: !2352)
!2371 = !DILocation(line: 406, column: 39, scope: !2352)
!2372 = !DILocation(line: 408, column: 45, scope: !2352)
!2373 = !DILocation(line: 408, column: 53, scope: !2352)
!2374 = !DILocation(line: 408, column: 13, scope: !2352)
!2375 = !DILocation(line: 409, column: 13, scope: !2352)
!2376 = !DILocation(line: 410, column: 45, scope: !2352)
!2377 = !DILocation(line: 410, column: 53, scope: !2352)
!2378 = !DILocation(line: 410, column: 13, scope: !2352)
!2379 = !DILocation(line: 411, column: 9, scope: !2352)
!2380 = !DILocation(line: 398, column: 42, scope: !2352)
!2381 = distinct !{!2381, !2358, !2379}
!2382 = !DILocation(line: 412, column: 5, scope: !2352)
!2383 = distinct !DISubprogram(name: "WriteVecElem<long long>", scope: !1, file: !1, line: 414, type: !16, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2384 = !DILocation(line: 414, column: 73, scope: !2383)
!2385 = !DILocation(line: 414, column: 81, scope: !2383)
!2386 = !DILocation(line: 414, column: 87, scope: !2383)
!2387 = !DILocation(line: 414, column: 94, scope: !2383)
!2388 = !DILocation(line: 414, column: 100, scope: !2383)
!2389 = !DILocation(line: 414, column: 59, scope: !2383)
!2390 = !DILocation(line: 414, column: 108, scope: !2383)
!2391 = !DILocation(line: 414, column: 119, scope: !2383)
!2392 = distinct !DISubprogram(name: "call<unsigned long long>", scope: !192, file: !192, line: 108, type: !16, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2393 = !DILocation(line: 108, column: 68, scope: !2392)
!2394 = !DILocation(line: 108, column: 61, scope: !2392)
!2395 = distinct !DISubprogram(name: "call<unsigned long long>", scope: !1, file: !1, line: 394, type: !16, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2396 = !DILocation(line: 394, column: 51, scope: !2395)
!2397 = !DILocation(line: 394, column: 66, scope: !2395)
!2398 = distinct !DISubprogram(name: "WriteArg<unsigned long long>", scope: !1, file: !1, line: 397, type: !16, scopeLine: 397, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2399 = !DILocation(line: 398, column: 18, scope: !2398)
!2400 = !DILocation(line: 398, column: 14, scope: !2398)
!2401 = !DILocation(line: 398, column: 28, scope: !2398)
!2402 = !DILocation(line: 398, column: 35, scope: !2398)
!2403 = !DILocation(line: 398, column: 33, scope: !2398)
!2404 = !DILocation(line: 398, column: 9, scope: !2398)
!2405 = !DILocation(line: 401, column: 17, scope: !2398)
!2406 = !DILocation(line: 401, column: 33, scope: !2398)
!2407 = !DILocation(line: 401, column: 30, scope: !2398)
!2408 = !DILocation(line: 401, column: 22, scope: !2398)
!2409 = !DILocation(line: 402, column: 40, scope: !2398)
!2410 = !DILocation(line: 402, column: 38, scope: !2398)
!2411 = !DILocation(line: 403, column: 41, scope: !2398)
!2412 = !DILocation(line: 403, column: 39, scope: !2398)
!2413 = !DILocation(line: 404, column: 13, scope: !2398)
!2414 = !DILocation(line: 405, column: 40, scope: !2398)
!2415 = !DILocation(line: 405, column: 38, scope: !2398)
!2416 = !DILocation(line: 406, column: 41, scope: !2398)
!2417 = !DILocation(line: 406, column: 39, scope: !2398)
!2418 = !DILocation(line: 408, column: 45, scope: !2398)
!2419 = !DILocation(line: 408, column: 53, scope: !2398)
!2420 = !DILocation(line: 408, column: 13, scope: !2398)
!2421 = !DILocation(line: 409, column: 13, scope: !2398)
!2422 = !DILocation(line: 410, column: 45, scope: !2398)
!2423 = !DILocation(line: 410, column: 53, scope: !2398)
!2424 = !DILocation(line: 410, column: 13, scope: !2398)
!2425 = !DILocation(line: 411, column: 9, scope: !2398)
!2426 = !DILocation(line: 398, column: 42, scope: !2398)
!2427 = distinct !{!2427, !2404, !2425}
!2428 = !DILocation(line: 412, column: 5, scope: !2398)
!2429 = distinct !DISubprogram(name: "WriteVecElem<unsigned long long>", scope: !1, file: !1, line: 414, type: !16, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2430 = !DILocation(line: 414, column: 73, scope: !2429)
!2431 = !DILocation(line: 414, column: 81, scope: !2429)
!2432 = !DILocation(line: 414, column: 87, scope: !2429)
!2433 = !DILocation(line: 414, column: 94, scope: !2429)
!2434 = !DILocation(line: 414, column: 100, scope: !2429)
!2435 = !DILocation(line: 414, column: 59, scope: !2429)
!2436 = !DILocation(line: 414, column: 108, scope: !2429)
!2437 = !DILocation(line: 414, column: 119, scope: !2429)
!2438 = distinct !DISubprogram(name: "call<double>", scope: !192, file: !192, line: 108, type: !16, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2439 = !DILocation(line: 108, column: 68, scope: !2438)
!2440 = !DILocation(line: 108, column: 61, scope: !2438)
!2441 = distinct !DISubprogram(name: "call<double>", scope: !1, file: !1, line: 394, type: !16, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2442 = !DILocation(line: 394, column: 51, scope: !2441)
!2443 = !DILocation(line: 394, column: 66, scope: !2441)
!2444 = distinct !DISubprogram(name: "WriteArg<double>", scope: !1, file: !1, line: 397, type: !16, scopeLine: 397, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2445 = !DILocation(line: 398, column: 18, scope: !2444)
!2446 = !DILocation(line: 398, column: 14, scope: !2444)
!2447 = !DILocation(line: 398, column: 28, scope: !2444)
!2448 = !DILocation(line: 398, column: 35, scope: !2444)
!2449 = !DILocation(line: 398, column: 33, scope: !2444)
!2450 = !DILocation(line: 398, column: 9, scope: !2444)
!2451 = !DILocation(line: 401, column: 17, scope: !2444)
!2452 = !DILocation(line: 401, column: 33, scope: !2444)
!2453 = !DILocation(line: 401, column: 30, scope: !2444)
!2454 = !DILocation(line: 401, column: 22, scope: !2444)
!2455 = !DILocation(line: 402, column: 40, scope: !2444)
!2456 = !DILocation(line: 402, column: 38, scope: !2444)
!2457 = !DILocation(line: 403, column: 41, scope: !2444)
!2458 = !DILocation(line: 403, column: 39, scope: !2444)
!2459 = !DILocation(line: 404, column: 13, scope: !2444)
!2460 = !DILocation(line: 405, column: 40, scope: !2444)
!2461 = !DILocation(line: 405, column: 38, scope: !2444)
!2462 = !DILocation(line: 406, column: 41, scope: !2444)
!2463 = !DILocation(line: 406, column: 39, scope: !2444)
!2464 = !DILocation(line: 408, column: 45, scope: !2444)
!2465 = !DILocation(line: 408, column: 53, scope: !2444)
!2466 = !DILocation(line: 408, column: 13, scope: !2444)
!2467 = !DILocation(line: 409, column: 13, scope: !2444)
!2468 = !DILocation(line: 410, column: 45, scope: !2444)
!2469 = !DILocation(line: 410, column: 53, scope: !2444)
!2470 = !DILocation(line: 410, column: 13, scope: !2444)
!2471 = !DILocation(line: 411, column: 9, scope: !2444)
!2472 = !DILocation(line: 398, column: 42, scope: !2444)
!2473 = distinct !{!2473, !2450, !2471}
!2474 = !DILocation(line: 412, column: 5, scope: !2444)
!2475 = distinct !DISubprogram(name: "WriteVecElem<double>", scope: !1, file: !1, line: 414, type: !16, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2476 = !DILocation(line: 414, column: 73, scope: !2475)
!2477 = !DILocation(line: 414, column: 81, scope: !2475)
!2478 = !DILocation(line: 414, column: 87, scope: !2475)
!2479 = !DILocation(line: 414, column: 94, scope: !2475)
!2480 = !DILocation(line: 414, column: 100, scope: !2475)
!2481 = !DILocation(line: 414, column: 59, scope: !2475)
!2482 = !DILocation(line: 414, column: 108, scope: !2475)
!2483 = !DILocation(line: 414, column: 119, scope: !2475)
!2484 = distinct !DISubprogram(name: "call<void *>", scope: !192, file: !192, line: 108, type: !16, scopeLine: 108, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2485 = !DILocation(line: 108, column: 68, scope: !2484)
!2486 = !DILocation(line: 108, column: 61, scope: !2484)
!2487 = distinct !DISubprogram(name: "call<void *>", scope: !1, file: !1, line: 394, type: !16, scopeLine: 394, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2488 = !DILocation(line: 394, column: 51, scope: !2487)
!2489 = !DILocation(line: 394, column: 66, scope: !2487)
!2490 = distinct !DISubprogram(name: "WriteArg<void *>", scope: !1, file: !1, line: 397, type: !16, scopeLine: 397, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2491 = !DILocation(line: 398, column: 18, scope: !2490)
!2492 = !DILocation(line: 398, column: 14, scope: !2490)
!2493 = !DILocation(line: 398, column: 28, scope: !2490)
!2494 = !DILocation(line: 398, column: 35, scope: !2490)
!2495 = !DILocation(line: 398, column: 33, scope: !2490)
!2496 = !DILocation(line: 398, column: 9, scope: !2490)
!2497 = !DILocation(line: 401, column: 17, scope: !2490)
!2498 = !DILocation(line: 401, column: 33, scope: !2490)
!2499 = !DILocation(line: 401, column: 30, scope: !2490)
!2500 = !DILocation(line: 401, column: 22, scope: !2490)
!2501 = !DILocation(line: 402, column: 40, scope: !2490)
!2502 = !DILocation(line: 402, column: 38, scope: !2490)
!2503 = !DILocation(line: 403, column: 41, scope: !2490)
!2504 = !DILocation(line: 403, column: 39, scope: !2490)
!2505 = !DILocation(line: 404, column: 13, scope: !2490)
!2506 = !DILocation(line: 405, column: 40, scope: !2490)
!2507 = !DILocation(line: 405, column: 38, scope: !2490)
!2508 = !DILocation(line: 406, column: 41, scope: !2490)
!2509 = !DILocation(line: 406, column: 39, scope: !2490)
!2510 = !DILocation(line: 408, column: 45, scope: !2490)
!2511 = !DILocation(line: 408, column: 53, scope: !2490)
!2512 = !DILocation(line: 408, column: 13, scope: !2490)
!2513 = !DILocation(line: 409, column: 13, scope: !2490)
!2514 = !DILocation(line: 410, column: 45, scope: !2490)
!2515 = !DILocation(line: 410, column: 53, scope: !2490)
!2516 = !DILocation(line: 410, column: 13, scope: !2490)
!2517 = !DILocation(line: 411, column: 9, scope: !2490)
!2518 = !DILocation(line: 398, column: 42, scope: !2490)
!2519 = distinct !{!2519, !2496, !2517}
!2520 = !DILocation(line: 412, column: 5, scope: !2490)
!2521 = distinct !DISubprogram(name: "WriteVecElem<void *>", scope: !1, file: !1, line: 414, type: !16, scopeLine: 414, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!2522 = !DILocation(line: 414, column: 73, scope: !2521)
!2523 = !DILocation(line: 414, column: 81, scope: !2521)
!2524 = !DILocation(line: 414, column: 87, scope: !2521)
!2525 = !DILocation(line: 414, column: 94, scope: !2521)
!2526 = !DILocation(line: 414, column: 100, scope: !2521)
!2527 = !DILocation(line: 414, column: 59, scope: !2521)
!2528 = !DILocation(line: 414, column: 108, scope: !2521)
!2529 = !DILocation(line: 414, column: 119, scope: !2521)
