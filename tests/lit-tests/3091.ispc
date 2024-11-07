// RUN: %{ispc} %s --target=avx2-i32x8 --target-os=windows --vectorcall --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && WINDOWS_ENABLED

// CHECK-NOT: unreachable
export uniform float Round(uniform float x) { return round(x); }
export uniform float Log(uniform float x)   { return log(x); }