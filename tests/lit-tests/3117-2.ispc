// RUN: %{ispc} --target=neon-i32x8 --nowrap --enable-llvm-intrinsics --emit-llvm-text %s -o - 2>&1 | FileCheck %s

// REQUIRES: ARM_ENABLED

// CHECK: define void @mask_type___(<[[WIDTH:.*]] x [[MASK:.*]]> %__mask)
void mask_type() {}

// CHECK-LABEL: @foo
// CHECK-DAG: %calltmp = tail call <[[WIDTH]] x i16> @llvm.aarch64.neon.sqrdmulh.v[[WIDTH]]i16(<[[WIDTH]] x i16> %t, <[[WIDTH]] x i16> %t)

// CHECK-NOT: Intrinsic name not mangled correctly for type arguments! Should be: llvm.aarch64.neon.sqrdmulh.v8i16
int16 foo(int16 t) {
    return @llvm.aarch64.neon.sqrdmulh(t, t);
}
