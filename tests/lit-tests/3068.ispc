//; RUN: %{ispc} %s --target=avx2-i32x8 --vectorcall --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && WINDOWS_ENABLED

// CHECK-NOT: unreachable
float foo(float x) { return round(x); }