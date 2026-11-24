// #3804: `inline` used to get LLVM `alwaysinline` at parse time, forcing the
// first inliner to expand callees before they were optimized. Now we emit a
// deferred marker instead, converted to `alwaysinline` only before the final
// inliner (see restore-inline-attr.ll, ispc-opt.ll).

// Frontend must emit the marker, not enum alwaysinline. Catches reintroducing
// early alwaysinline (which would print `{ alwaysinline nounwind ... }`).
// RUN: %{ispc} %s --target=host --nowrap --nostdlib --emit-llvm-text \
// RUN:   --debug-phase=first:first -o - | FileCheck %s --check-prefix=PREOPT
// PREOPT: define internal {{.*}}@helper{{.*}}#[[ATTRS:[0-9]+]]
// PREOPT: attributes #[[ATTRS]] = { nounwind "ispc-defer-alwaysinline"

// ...but `inline` still gets inlined away by the end.
// RUN: %{ispc} %s --target=host --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefix=FINAL --implicit-check-not='@helper'
// FINAL-LABEL: define {{.*}} @caller
// FINAL: ret

static inline uniform int helper(uniform int x) {
    return x * 3 + 1;
}

uniform int caller(uniform int x) {
    return helper(x) + helper(x + 7);
}
