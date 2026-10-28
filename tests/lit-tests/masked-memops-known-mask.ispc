// Pins how ImproveMemoryOps treats masked loads and stores whose mask is known:
// an all-on masked load becomes an ordinary load, and an all-off masked store
// is removed. The checks describe today's output; each case gives the ideal
// output in an "Ideal:" comment.

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefix=FINAL
// RUN: rm -rf %t && %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text \
// RUN:   --debug-phase=211:211 --dump-file=%t -o /dev/null
// RUN: FileCheck %s --check-prefix=PHASE --input-file=%t/ir_211_ispcImproveMemoryOpsPass.ll

// REQUIRES: X86_ENABLED

// Consecutive doubles: the gather becomes a masked load, and in the all-on
// path of the function that masked load becomes one 64-byte load. A masked
// load that stays masked is two 32-byte AVX loads instead.
// Ideal: = today.
// FINAL-LABEL: @all_on_load___
// FINAL: load <8 x double>, ptr {{%.*}}, align 8
void all_on_load(uniform double p[], uniform double r[]) { r[programIndex] = p[programIndex + 1]; }

// The condition is false in every lane, so the store's mask is all off. The
// first ImproveMemoryOps run (the phase dumped above) removes the store; later
// passes would not always clean up the code around it (for example inside
// fmod).
// Ideal: = today.
// PHASE-LABEL: @all_off_store___
// PHASE-NOT: {{@__pseudo_masked_store|@__pseudo_scatter|^ +store }}
// PHASE: {{^[}]$}}
void all_off_store(uniform float p[]) {
    if (programIndex > 1000)
        p[programIndex] = 1;
}
