// sizeof of a bool type is its size in memory, the distance between two
// elements of an array, which uniform pointer arithmetic and the C header
// use too: one byte per lane for a varying bool, one byte per element for a
// bool short vector. It used to be the size of the register type, the mask:
// 32 bytes for a varying bool on avx2-i32x8, 2 on avx512skx-x16, and 1 for a
// uniform bool<4>.

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=AVX2
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=SKX

// REQUIRES: X86_ENABLED

// AVX2-LABEL: define {{.*}}@varying_bool
// AVX2: ret i32 8
// SKX-LABEL: define {{.*}}@varying_bool
// SKX: ret i32 16
uniform int varying_bool() { return sizeof(varying bool); }

// AVX2-LABEL: define {{.*}}@varying_bool_stride
// AVX2: ret i32 8
// SKX-LABEL: define {{.*}}@varying_bool_stride
// SKX: ret i32 16
uniform int varying_bool_stride(varying bool *uniform p) {
    return (uniform int)((uniform int8 * uniform)(p + 1) - (uniform int8 * uniform)p);
}

// AVX2-LABEL: define {{.*}}@varying_bool4
// AVX2: ret i32 32
// SKX-LABEL: define {{.*}}@varying_bool4
// SKX: ret i32 64
uniform int varying_bool4() { return sizeof(varying bool<4>); }

// AVX2-LABEL: define {{.*}}@uniform_bool4
// AVX2: ret i32 4
// SKX-LABEL: define {{.*}}@uniform_bool4
// SKX: ret i32 4
uniform int uniform_bool4() { return sizeof(uniform bool<4>); }

// AVX2-LABEL: define {{.*}}@uniform_bool4_stride
// AVX2: ret i32 4
// SKX-LABEL: define {{.*}}@uniform_bool4_stride
// SKX: ret i32 4
uniform int uniform_bool4_stride(uniform bool<4> *uniform p) {
    return (uniform int)((uniform int8 * uniform)(p + 1) - (uniform int8 * uniform)p);
}

// A constant sizeof, here an array size, is the same.
// AVX2-LABEL: define {{.*}}@array_size
// AVX2: ret i32 8
// SKX-LABEL: define {{.*}}@array_size
// SKX: ret i32 16
uniform int array_size() {
    uniform int8 bytes[sizeof(varying bool)];
    return sizeof(bytes);
}
