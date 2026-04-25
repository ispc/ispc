// Regression test for #3804: `inline`-qualified user functions used to be
// inlined too early in the optimization pipeline, hurting code quality. The
// parse-time marker is now `ispc-defer-alwaysinline`, which RestoreInlineAttrPass
// converts to `alwaysinline` shortly before the final inliner pass runs. By
// the end of optimization the callee is fully inlined and removed.

// RUN: %{ispc} %s --target=host --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --implicit-check-not='@helper'

// Verify the callee is inlined and erased: only `caller` remains in the
// final IR — no standalone `helper` definition and no call to `helper`
// from inside `caller`. --implicit-check-not enforces this across the
// whole input, including inside `caller`'s body.
// CHECK-LABEL: define {{.*}} @caller
// CHECK: ret

static inline uniform int helper(uniform int x) {
    return x * 3 + 1;
}

uniform int caller(uniform int x) {
    return helper(x) + helper(x + 7);
}
