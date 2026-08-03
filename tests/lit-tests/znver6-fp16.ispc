// Zen6 adds AVX512-FP16 on top of Zen5, so float16 arithmetic must lower to
// native half-precision instructions instead of round-tripping through fp32.
// The znver5 run is the negative control: same target, so any difference comes
// from the CPU alone.

// RUN: %{ispc} -O2 %s --target=avx512icl-x16 --cpu=znver6 --nostdlib --nowrap --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=ZNVER6 --implicit-check-not=vcvtph2ps
// RUN: %{ispc} -O2 %s --target=avx512icl-x16 --cpu=znver5 --nostdlib --nowrap --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=ZNVER5 --implicit-check-not='{{vfmadd[0-9]+ph}}'

// REQUIRES: X86_ENABLED && LLVM_23_0+

// ZNVER6: {{vfmadd[0-9]+ph}}
// ZNVER5: vcvtph2ps

void fma16(uniform float16 dst[], const uniform float16 a[], const uniform float16 b[], uniform int n) {
    foreach (i = 0 ... n) {
        dst[i] = a[i] * b[i] + dst[i];
    }
}
