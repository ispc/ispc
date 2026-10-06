// Zen6 adds AVX512-FP16 on top of Zen5, so float16 arithmetic must lower to
// native half-precision instructions instead of round-tripping through fp32.
// The znver5 run is the negative control: same target, so any difference comes
// from the CPU alone.
// The INTRINSIC runs check that the "avx512fp16" CPUFeatures entry lets an
// explicit @llvm.x86.avx512fp16.* intrinsic pass Target::checkIntrinsticSupport
// on znver6, while znver5 still rejects it.

// RUN: %{ispc} -O2 %s --target=avx512icl-x16 --cpu=znver6 --nostdlib --nowrap --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=ZNVER6 --implicit-check-not=vcvtph2ps
// RUN: %{ispc} -O2 %s --target=avx512icl-x16 --cpu=znver5 --nostdlib --nowrap --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=ZNVER5 --implicit-check-not='{{vfmadd[0-9]+ph}}'
// RUN: %{ispc} -O2 %s -DINTRINSIC --target=avx512icl-x16 --cpu=znver6 --enable-llvm-intrinsics --nostdlib --nowrap \
// RUN:   --emit-asm -o - | FileCheck %s --check-prefix=INTRINSIC-ZNVER6
// RUN: not %{ispc} -O2 %s -DINTRINSIC --target=avx512icl-x16 --cpu=znver5 --enable-llvm-intrinsics --nostdlib \
// RUN:   --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=INTRINSIC-ZNVER5

// REQUIRES: X86_ENABLED && LLVM_23_0+

// ZNVER6: {{vfmadd[0-9]+ph}}
// ZNVER5: vcvtph2ps

void fma16(uniform float16 dst[], const uniform float16 a[], const uniform float16 b[], uniform int n) {
    foreach (i = 0 ... n) {
        dst[i] = a[i] * b[i] + dst[i];
    }
}

#ifdef INTRINSIC
// INTRINSIC-ZNVER6-LABEL: max16___
// INTRINSIC-ZNVER6: vmaxph
// INTRINSIC-ZNVER5: Error: Target specific LLVM intrinsic "avx512fp16.max.ph.256" not supported on "znver5" CPU.
float16 max16(float16 a, float16 b) { return @llvm.x86.avx512fp16.max.ph.256(a, b); }
#endif
