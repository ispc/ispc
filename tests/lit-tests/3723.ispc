// RUN: %{ispc} %s -O0 --target=sse4,avx10.2dmr-x16 --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// RUN: FileCheck --input-file=%t_avx10_2dmr.ll %s

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_20_0+

// CHECK-NOT: @function_avx10.2dmr
// CHECK: @function_avx10_2dmr

export void function() {}
