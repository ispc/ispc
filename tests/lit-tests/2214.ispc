// RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i64x4
// RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i32x8
// RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i16x16
// RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i8x32
// RUN: %{ispc} %s -o %t.o --nostdlib --target=host

// REQUIRES: X86_ENABLED

// The key here is to tests launch expression with function pointer on targets with different base type (i8-i64).

typedef task void (*TaskFn)(float f);

task void x(float f) {
}

void f_f(float f) {
    uniform TaskFn func = x;
    launch [10000] func(f);
    sync;
    launch [1][2][3] func(f);
    sync;
}

