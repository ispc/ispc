// Check that LLVM intrinsics can be called with template-dependent argument types.
// The intrinsic declaration is deferred until the template is instantiated.

// RUN: %{ispc} --target=avx2-i32x8 --nowrap --enable-llvm-intrinsics --emit-llvm-text %s -o - 2>&1 | FileCheck %s
// RUN: not %{ispc} --target=avx2-i32x8 --nowrap -DNO_INSTANCE %s -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED
// RUN: not %{ispc} --target=avx2-i32x8 --nowrap --enable-llvm-intrinsics -DNO_INSTANCE -DBAD_NAME %s -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-NAME

// REQUIRES: X86_ENABLED

// CHECK-NOT: FATAL ERROR
// CHECK-DISABLED: Calling LLVM intrinsics from ISPC source code is an experimental feature
// CHECK-DISABLED-NOT: FATAL ERROR

// Unknown intrinsic names are reported even if the template is never instantiated.
// CHECK-BAD-NAME: Error: LLVM intrinsic "llvm.typo" not supported.
// CHECK-BAD-NAME: Error: LLVM intrinsic "llvm.ispc.typo" not supported.
// CHECK-BAD-NAME-NOT: FATAL ERROR

// Never instantiated.
template <typename T> T unused(T x) { return @llvm.ctlz(x, false); }

#ifdef BAD_NAME
template <typename T> T bad_llvm(T x) { return @llvm.typo(x); }
template <typename T> T bad_ispc(T x) { return @llvm.ispc.typo(x); }
#endif

#ifndef NO_INSTANCE
template <typename T> T cttz(T x) { return @llvm.cttz(x, false); }

template <typename T> T ctpop(T x) {
    T y = x;
    return @llvm.ctpop(y);
}

template <typename T> T abs(T x) { return @llvm.abs(x, false); }

// Nested template with a dependent argument.
template <typename T> T outer(T x) { return abs<T>(x) + @llvm.cttz(x, true); }

// CHECK-LABEL: define {{.*}} @test_cttz___
// CHECK: call {{.*}}i32 @llvm.cttz.i32(i32 %x, i1 false)
uniform uint32 test_cttz(uniform uint32 x) { return cttz<uniform uint32>(x); }

// Each instantiation gets its own declaration, also with deduced template arguments.
// CHECK-LABEL: define {{.*}} @test_cttz_deduced___
// CHECK: call {{.*}}<8 x i8> @llvm.cttz.v8i8(<8 x i8> %x, i1 false)
varying int8 test_cttz_deduced(varying int8 x) { return cttz(x); }

// CHECK-LABEL: define {{.*}} @test_ctpop___
// CHECK: call {{.*}}<8 x i64> @llvm.ctpop.v8i64(<8 x i64>
varying int64 test_ctpop(varying int64 x) { return ctpop<varying int64>(x); }

// CHECK-LABEL: define {{.*}} @test_abs___
// CHECK: call <8 x i32> @llvm.abs.v8i32(<8 x i32>
varying int32 test_abs(varying int32 x) { return abs<varying int32>(x); }

// CHECK-LABEL: define {{.*}} @test_outer___
// CHECK-DAG: call i64 @llvm.abs.i64(i64 %x, i1 false)
// CHECK-DAG: call {{.*}}i64 @llvm.cttz.i64(i64 %x, i1 true)
uniform int64 test_outer(uniform int64 x) { return outer<uniform int64>(x); }
#endif
