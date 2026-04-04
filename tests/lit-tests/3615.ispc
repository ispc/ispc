// RUN: %{ispc} --emit-llvm-text  -O3 --enable-llvm-intrinsics --target=avx512icl-x4 %s -o - | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} --emit-llvm-text  -O3 --enable-llvm-intrinsics --target=avx2-i32x8 %s -o - | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc} --emit-llvm-text  -O3 --enable-llvm-intrinsics --target=avx2-i8x32 %s -o - | FileCheck %s -check-prefix=CHECK_AVX2_I8X32
// RUN: %{ispc} --emit-llvm-text  -O3 --enable-llvm-intrinsics --target=avx512skx-x32 %s -o - | FileCheck %s -check-prefix=CHECK_AVX512SKX_X32
// RUN: %{ispc} --emit-llvm-text  -O3 --enable-llvm-intrinsics --target=avx512skx-x64 %s -o - | FileCheck %s -check-prefix=CHECK_AVX512SKX_X64
// REQUIRES: X86_ENABLED

// CHECK_AVX512-LABEL: @popcnt_ispc_32___vyi
// CHECK_AVX512: call{{.*}}<4 x i32> @llvm.ctpop.v4i32
// CHECK_AVX2-LABEL: @popcnt_ispc_32___vyi
// CHECK_AVX2: call{{.*}}i32 @llvm.ctpop.i32
// CHECK_AVX2_I8X32-LABEL: @popcnt_ispc_32___vyi
// CHECK_AVX2_I8X32: call{{.*}}<32 x i32> @llvm.ctpop.v32i32
// CHECK_AVX512SKX_X32-LABEL: @popcnt_ispc_32___vyi
// CHECK_AVX512SKX_X32: call{{.*}}<32 x i32> @llvm.ctpop.v32i32
// CHECK_AVX512SKX_X64-LABEL: @popcnt_ispc_32___vyi
// CHECK_AVX512SKX_X64: call{{.*}}<64 x i32> @llvm.ctpop.v64i32

varying int popcnt_ispc_32(varying int num) {
    return popcnt(num);
}

// CHECK_AVX512-LABEL: @popcnt_ispc_64___vyI
// CHECK_AVX512: call{{.*}}<4 x i64> @llvm.ctpop.v4i64
// CHECK_AVX2-LABEL: @popcnt_ispc_64___vyI
// CHECK_AVX2: call{{.*}}i64 @llvm.ctpop.i64
// CHECK_AVX2_I8X32-LABEL: @popcnt_ispc_64___vyI
// CHECK_AVX2_I8X32: call{{.*}}<32 x i64> @llvm.ctpop.v32i64
// CHECK_AVX512SKX_X32-LABEL: @popcnt_ispc_64___vyI
// CHECK_AVX512SKX_X32: call{{.*}}<32 x i64> @llvm.ctpop.v32i64
// CHECK_AVX512SKX_X64-LABEL: @popcnt_ispc_64___vyI
// CHECK_AVX512SKX_X64: call{{.*}}<64 x i64> @llvm.ctpop.v64i64

varying int64 popcnt_ispc_64(varying int64 num) {
    return popcnt(num);
}
