// RUN: %{ispc} %s -g -O2 -o %t.o --target=avx512skx-x16
// RUN: %{ispc} %s -g -O2 -o %t.o --target=avx512knl-x16

// REQUIRES: X86_ENABLED

export void add_lists(const uniform float a[], const uniform float b[],
        uniform float c[], const uniform int count) {
    foreach (i = 0 ... count){
        c[i] = a[i] + b[i];
    }
}
