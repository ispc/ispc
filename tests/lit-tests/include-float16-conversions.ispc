// This test checks that --include-float16-conversions brings half conversion
// functions to the compiled module.

// RUN: %{ispc} --target=host --nowrap --include-float16-conversions --emit-llvm-text %s -o - 2>&1 | FileCheck %s

// CHECK: __truncsfhf2
// CHECK: __extendhfsf2
// CHECK: __truncdfhf2
// CHECK: __extendhfdf2

int foo();
