// RUN: %{ispc} --pic --target=avx2-i8x32 %s -O2 --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: store i64 496, {{.*}} %print_arg

extern "C" uniform int main() {
  print("%\n", reduce_add(programIndex));
  return 0;
}
