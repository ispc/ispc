// RUN: %{ispc} -O0 --target=avx2-i32x8 %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// CHECK-NOT: Gather

#define REAL float

void translate (uniform REAL vector[], uniform REAL x[], uniform size_t size)
{
  foreach (i = 0 ... size) {
    x[i] += vector[0];
  }
}

