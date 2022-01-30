// RUN: %{ispc} --target=avx1-i32x4 --cpu=ps4 --arch=x86-64 -g -O0  %s -o %t.o
// RUN: %{ispc} --target=avx1-i32x4 --cpu=haswell --arch=x86-64 -g -O0  %s -o %t.o

// REQUIRES: X86_ENABLED

// Should compile without a crash

double b(double c, int d) {
  return shuffle(c, d);
}
