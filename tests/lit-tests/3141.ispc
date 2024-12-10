// RUN: not %{ispc} --nowrap --target=host --enable-llvm-intrinsics --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// REQUIRES: LLVM_17_0+

#if LLVM_VERSION_MAJOR >= 19
#define DEINTERLEAVE2 llvm.vector.deinterleave2
#else
#define DEINTERLEAVE2 llvm.experimental.vector.deinterleave2
#endif

// This checks that logic for deducting LLVM type for overloaded LLVM
// intrinsics works.
// CHECK-NOT: FATAL ERROR

// This error is expected because the logic of mapping LLVM function signatures
// to ISPC ones does not support the struct return type.
// CHECK: Error: Return type not representable for Intrinsic
void __deinterleave2(uniform float<8> val) {
    @DEINTERLEAVE2(val);
}
