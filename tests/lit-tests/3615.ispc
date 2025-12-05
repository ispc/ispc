// RUN: %{ispc} --emit-llvm-text  -O3 --enable-llvm-intrinsics --target=avx512icl-x4 %s -o - | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} --emit-llvm-text  -O3 --enable-llvm-intrinsics --target=avx2-i32x8 %s -o - | FileCheck %s -check-prefix=CHECK_AVX2
// REQUIRES: X86_ENABLED

// CHECK_AVX512-LABEL: @popcnt_ispc_32___vyi
// CHECK_AVX512: call{{.*}}<4 x i32> @llvm.ctpop.v4i32
// CHECK_AVX2-LABEL: @popcnt_ispc_32___vyi
// CHECK_AVX2: call{{.*}}i32 @llvm.ctpop.i32

varying int popcnt_ispc_32(varying int num) {
    return popcnt(num);
}

// CHECK_AVX512-LABEL: @popcnt_ispc_64___vyI
// CHECK_AVX512: call{{.*}}<4 x i64> @llvm.ctpop.v4i64
// CHECK_AVX2-LABEL: @popcnt_ispc_64___vyI
// CHECK_AVX2: call{{.*}}i64 @llvm.ctpop.i64

varying int64 popcnt_ispc_64(varying int64 num) {
    return popcnt(num);
}
