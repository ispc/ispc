// RUN: %{ispc} --target=avx2-i32x16,avx512skx-x16 %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// REQUIRES: X86_ENABLED
// CHECK-NOT: Please file a bug
void foo( uniform int  a, uniform int b[]) {
    b[0] = a;
}
