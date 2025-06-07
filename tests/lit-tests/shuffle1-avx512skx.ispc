// Test shuffle asm codegen for skx targets

// Shuffle Instruction Patterns by Data Type and Width
//
// int8 (1-byte integers)
//
// - x4, x8, x16: vpshufb - Byte-level shuffle within 128/256-bit registers
// - x32: vpermw - Transitions to word-level permutation for larger vectors
// - x64: vpermi2w - Uses indexed word permutation for maximum width
//
// int16/float16 (2-byte integers/half-precision floats)
//
// - x4, x8, x16, x32: vpermw - Consistent word-level permutation
// - x64: vpermi2w - Indexed word permutation for 64-wide vectors
//
// int32/float (4-byte integers/single-precision floats)
//
// - x4: vpermilps - Simple lane permutation within 128-bit registers
// - x8, x16: vpermps - Packed single permutation for 256/512-bit
// - x32: vpermi2d - Indexed double-word permutation
// - x64: vpgatherdd - Memory gather operation for very wide vectors
//
// int64/double (8-byte integers/double-precision floats)
//
// - x4: vpermq - Quad-word permutation within 256-bit registers
// - x8, x16: vpermi2q - Indexed quad-word permutation
// - x32, x64: vpgatherdq - Memory gather operations for wide double-precision vectors
//
// Instruction Evolution Pattern
//
// The instruction selection follows a clear optimization strategy:
//
// 1. Small widths (x4-x8): Use direct permutation instructions optimized for the data type size
// 2. Medium widths (x16-x32): Transition to indexed permutation instructions (vpermi2*)
// 3. Large widths (x64): Use memory gather instructions (vpgather*) when vector becomes too wide for efficient register-based shuffling

// RUN: %{ispc} %s --target=avx512skx-x4 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X4
// RUN: %{ispc} %s --target=avx512skx-x8 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X8
// RUN: %{ispc} %s --target=avx512skx-x16 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X16
// RUN: %{ispc} %s --target=avx512skx-x32 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X32
// RUN: %{ispc} %s --target=avx512skx-x64 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X64

// REQUIRES: X86_ENABLED

// CHECK-X4-LABEL: shuffle_i8___vytvyi:
// CHECK-X4-DAG: vpshufb
// CHECK-X4-NOT: vpextrd
// CHECK-X4-NOT: vpinsrb
// CHECK-X8-LABEL: shuffle_i8___vytvyi:
// CHECK-X8-DAG: vpshufb
// CHECK-X8-NOT: vpextrd
// CHECK-X8-NOT: vpinsrb
// CHECK-X16-LABEL: shuffle_i8___vytvyi:
// CHECK-X16-DAG: vpshufb
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrb
// CHECK-X32-LABEL: shuffle_i8___vytvyi:
// CHECK-X32-DAG: vpermw
// CHECK-X32-NOT: vpextrd
// CHECK-X32-NOT: vpinsrb
// CHECK-X64-LABEL: shuffle_i8___vytvyi:
// CHECK-X64-DAG: vpermi2w
// CHECK-X64-NOT: vpextrd
// CHECK-X64-NOT: vpinsrb
int8 shuffle_i8(int8 x, int32 perm) {
    return shuffle(x, perm);
}

// CHECK-X4-LABEL: shuffle_i16___vysvyi:
// CHECK-X4-DAG: vpermw
// CHECK-X4-NOT: vpextrd
// CHECK-X4-NOT: vpinsrw
// CHECK-X8-LABEL: shuffle_i16___vysvyi:
// CHECK-X8-DAG: vpermw
// CHECK-X8-NOT: vpextrd
// CHECK-X8-NOT: vpinsrw
// CHECK-X16-LABEL: shuffle_i16___vysvyi:
// CHECK-X16-DAG: vpermw
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrw
// CHECK-X32-LABEL: shuffle_i16___vysvyi:
// CHECK-X32-DAG: vpermw
// CHECK-X32-NOT: vpextrd
// CHECK-X32-NOT: vpinsrw
// CHECK-X64-LABEL: shuffle_i16___vysvyi:
// CHECK-X64-DAG: vpermi2w
// CHECK-X64-NOT: vpextrd
// CHECK-X64-NOT: vpinsrw
int16 shuffle_i16(int16 x, int32 perm) {
    return shuffle(x, perm);
}

// CHECK-X4-LABEL: shuffle_i32___vyivyi:
// CHECK-X4-DAG: vpermilps
// CHECK-X8-LABEL: shuffle_i32___vyivyi:
// CHECK-X8-DAG: vpermps
// CHECK-X16-LABEL: shuffle_i32___vyivyi:
// CHECK-X16-DAG: vpermps
// CHECK-X32-LABEL: shuffle_i32___vyivyi:
// CHECK-X32-DAG: vpermi2d
// CHECK-X64-LABEL: shuffle_i32___vyivyi:
// CHECK-X64-DAG: vpgatherdd
int32 shuffle_i32(int32 x, int32 perm) {
    return shuffle(x, perm);
}

// CHECK-X4-LABEL: shuffle_i64___vyIvyi:
// CHECK-X4-DAG: vpermq
// CHECK-X8-LABEL: shuffle_i64___vyIvyi:
// CHECK-X8-DAG: vpermi2q
// CHECK-X8-NOT: vpextrd
// CHECK-X16-LABEL: shuffle_i64___vyIvyi:
// CHECK-X16-DAG: vpermi2q
// CHECK-X16-NOT: vpextrd
// CHECK-X32-LABEL: shuffle_i64___vyIvyi:
// CHECK-X32-DAG: vpgatherdq
// CHECK-X32-NOT: vpextrd
// CHECK-X64-LABEL: shuffle_i64___vyIvyi:
// CHECK-X64-DAG: vpgatherdq
// CHECK-X64-NOT: vpextrd
int64 shuffle_i64(int64 x, int32 perm) {
    return shuffle(x, perm);
}

// CHECK-X4-LABEL: shuffle_half___vyhvyi:
// CHECK-X4-DAG: vpermw
// CHECK-X4-NOT: vpextrd
// CHECK-X4-NOT: vpinsrw
// CHECK-X8-LABEL: shuffle_half___vyhvyi:
// CHECK-X8-DAG: vpermw
// CHECK-X8-NOT: vpextrd
// CHECK-X8-NOT: vpinsrw
// CHECK-X16-LABEL: shuffle_half___vyhvyi:
// CHECK-X16-DAG: vpermw
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrw
// CHECK-X32-LABEL: shuffle_half___vyhvyi:
// CHECK-X32-DAG: vpermw
// CHECK-X32-NOT: vpextrd
// CHECK-X32-NOT: vpinsrw
// CHECK-X64-LABEL: shuffle_half___vyhvyi:
// CHECK-X64-DAG: vpermi2w
// CHECK-X64-NOT: vpextrd
// CHECK-X64-NOT: vpinsrw
float16 shuffle_half(float16 x, int32 perm) {
    return shuffle(x, perm);
}

// CHECK-X4-LABEL: shuffle_float___vyfvyi:
// CHECK-X4-DAG: vpermilps
// CHECK-X8-LABEL: shuffle_float___vyfvyi:
// CHECK-X8-DAG: vpermps
// CHECK-X16-LABEL: shuffle_float___vyfvyi:
// CHECK-X16-DAG: vpermps
// CHECK-X32-LABEL: shuffle_float___vyfvyi:
// CHECK-X32-DAG: vpermi2ps
// CHECK-X64-LABEL: shuffle_float___vyfvyi:
// CHECK-X64-DAG: vpgatherdd
float shuffle_float(float x, int32 perm) {
    return shuffle(x, perm);
}

// CHECK-X4-LABEL: shuffle_double___vydvyi:
// CHECK-X4-DAG: vpermq
// CHECK-X8-LABEL: shuffle_double___vydvyi:
// CHECK-X8-DAG: vpermi2pd
// CHECK-X16-LABEL: shuffle_double___vydvyi:
// CHECK-X16-DAG: vpermi2pd
// CHECK-X32-LABEL: shuffle_double___vydvyi:
// CHECK-X32-DAG: vpgatherdq
// CHECK-X64-LABEL: shuffle_double___vydvyi:
// CHECK-X64-DAG: vpgatherdq
double shuffle_double(double x, int32 perm) {
    return shuffle(x, perm);
}
