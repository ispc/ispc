// RUN: not %{ispc} --nowrap --target=host --enable-llvm-intrinsics --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// This checks that logic for deducting LLVM type for overloaded LLVM
// intrinsics works.
// CHECK-NOT: FATAL ERROR

// This error is expected because the logic of mapping LLVM function signatures
// to ISPC ones does not support the struct return type.
// CHECK: Return type "{ <4 x float>, <4 x float> }" of LLVM intrinsic "llvm.vector.deinterleave2" is not representable as an ISPC type.
void __deinterleave2(uniform float<8> val) {
    @llvm.vector.deinterleave2(val);
}
