// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s

// the order of BBs is not deterministic, so we need to check them with CHECK-DAG

// CHECK-LABEL: @foo(
// CHECK: foreach_full_body.lr.ph:
// CHECK: br i1 [[TMP:%.*]], label %[[FOREACH_SPLIT_US:.*]], label %[[FOREACH_SPLIT:.*]]
// CHECK-DAG: [[FOREACH_SPLIT_US]]:
// CHECK-DAG: [[FOREACH_SPLIT]]:
// CHECK-DAG: foreach_reset:
export void foo(const uniform uint32 input[], uniform uint32 output[],
                const uniform int index, const uniform int n)
{
    foreach (i = 0 ... n) {
        if (index == 0) {
                output[i] = input[i];
        } else {
                output[i] |= input[i];
        }
    }
}
