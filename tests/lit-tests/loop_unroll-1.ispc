// Checks that loop unrolling is not happening with O1 and when --opt-desable-loop-unrolling is used
// RUN: %{ispc} %s --target=host --nowrap -O1 --emit-llvm-text --no-discard-value-names --nostdlib -o - | FileCheck %s -check-prefix=CHECKO1
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --no-discard-value-names --nostdlib --opt=disable-loop-unroll -o - | FileCheck %s -check-prefix=CHECKO1
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --no-discard-value-names --nostdlib -o - | FileCheck %s -check-prefix=CHECKO2

// CHECK-LABEL: @compute
// CHECKO1-COUNT-1: @bar
// CHECKO2-COUNT-2: @bar
extern float bar(float);
void compute(uniform float input[], uniform float output[]) {
    foreach (i = 0 ... 16) {
        output[i] = bar(input[i]) + (input[i]);
    }
}
