// The size of a type whose element count depends on a template parameter is
// only known when the template is instantiated. As a constant, for example
// an array dimension, sizeof(uniform int<N>) used to give the size of the
// uninstantiated vector, padded to 4 elements: 16 bytes also for N = 8. It
// is an error now. In code, sizeof gives the instantiated size.

// RUN: not %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -DCONST -o %t.o 2>&1 | FileCheck %s --check-prefix=CONST
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=CODE

// REQUIRES: X86_ENABLED

#ifdef CONST
// CONST: Error: Array dimension must be a compile-time constant
template <int N> uniform int bytes() {
    uniform int8 b[sizeof(uniform int<N>)];
    return sizeof(b);
}
#else
// CODE-LABEL: define {{.*}}@run
// CODE: ret i32 32
template <int N> uniform int size() { return sizeof(uniform int<N>); }
#endif

#ifdef CONST
export uniform int run() { return bytes<8>(); }
#else
export uniform int run() { return size<8>(); }
#endif
