// RUN: %{ispc} --target=host --nostdlib --nowrap -o - --emit-llvm-text %s 2>&1 | FileCheck %s

// CHECK-NOT: Assertion failed
void foo(uniform int32* uniform bTbase, varying int64 v64) {
    foreach (i = 0 ... 100) {
        v64 = (&((int64*)(bTbase)))[i];
    }
}
