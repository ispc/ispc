// Test for https://github.com/ispc/ispc/issues/3748
// When passing a bool short-vector comparison result directly to a function,
// the LLVM type must match the function signature. Previously, the comparison
// produced [N x <W x iMASK>] but VectorType::LLVMType() for varying bool<N>
// returned [N x <W x i8>] (storage type), causing a type mismatch and a
// "Call parameter type does not match function signature" fatal error.
// Fixed by aligning lGetVectorLLVMType so that varying bool<N>::LLVMType()
// uses the mask element type (consistent with scalar varying bool).

// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: X86_ENABLED

// CHECK-LABEL: @foo___
// CHECK: fcmp oeq
// CHECK: fcmp oeq
// CHECK: fcmp oeq

inline varying bool all_elements(bool<3> v)
{
    return v.x && v.y && v.z;
}

// Previously crashed with: LLVM ERROR: Broken module found, compilation aborted!
// when passing a comparison result directly as a bool short-vector argument.
varying bool foo(float<3> b, float<3> c)
{
    return all_elements(b == c);
}
