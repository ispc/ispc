// Pins where the passes that lower gathers and scatters run relative to the
// inliner, from the --debug-pm output for one exported function k. The checks
// describe today's order; the ideal order is given in "Ideal:" comments, and
// the change that reaches it updates the checks.
//
// Only two kinds of lines are matched: module passes ("... on [module]") and
// function passes on exactly k ("... on k ("), not on its internal copy
// k___...
//
// Ideal at -O2: the final choice of the memory operation runs after the final
// inliner: ... -> ScalarizePass -> final inliner -> SROA, SimplifyCFG,
// InstCombine -> ImproveMemoryOps -> GatherCoalesce -> ReplacePseudoMemoryOps
// -> DCE.
// Ideal at -O0: lowering after the inliner.

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -O2 --debug-pm -o /dev/null 2>&1 | FileCheck %s --check-prefix=O2
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -O0 --debug-pm -o /dev/null 2>&1 | FileCheck %s --check-prefix=O0

// REQUIRES: X86_ENABLED

// O2-NOT: ModuleInlinerWrapperPass on [module]
// O2: ispc::ImproveMemoryOpsPass on k (
// O2: ModuleInlinerWrapperPass on [module]
// O2: ispc::ReplaceMaskedMemOpsPass on k (
// O2: ispc::ImproveMemoryOpsPass on k (
// O2: ispc::GatherCoalescePass on k (
// O2: ModuleInlinerWrapperPass on [module]
// O2: ispc::ImproveMemoryOpsPass on k (
// O2: ispc::ReplacePseudoMemoryOpsPass on k (
// O2: ModuleInlinerWrapperPass on [module]
// O2: ispc::ReplaceMaskedMemOpsPass on k (
// O2: ispc::IsCompileTimeConstantPass on k (
// O2: ispc::ScalarizePass on k (
// The final inliner: no memory operation pass between ScalarizePass and it,
// no other inliner and no memory operation pass after it.
// O2-NOT: ispc::{{ImproveMemoryOps|GatherCoalesce|ReplacePseudoMemoryOps|ReplaceMaskedMemOps}}Pass on k (
// O2: ModuleInlinerWrapperPass on [module]
// O2-NOT: ModuleInlinerWrapperPass on [module]
// O2-NOT: ispc::{{ImproveMemoryOps|GatherCoalesce|ReplacePseudoMemoryOps|ReplaceMaskedMemOps}}Pass on k (
// O2: GlobalDCEPass on [module]

// O0-NOT: ModuleInlinerWrapperPass on [module]
// O0: ispc::ImproveMemoryOpsPass on k (
// O0-NOT: ModuleInlinerWrapperPass on [module]
// O0: ispc::ReplacePseudoMemoryOpsPass on k (
// O0: ModuleInlinerWrapperPass on [module]
// O0-NOT: ModuleInlinerWrapperPass on [module]
// O0-NOT: ispc::{{ImproveMemoryOps|ReplacePseudoMemoryOps}}Pass on k (
// O0: GlobalDCEPass on [module]

export void k(uniform float *uniform f, uniform int *uniform i, uniform float *uniform r) {
    r[programIndex] = f[i[programIndex]];
}
