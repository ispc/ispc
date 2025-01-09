// Checks that function inlining is not happening with O1 and the function has optsize attribute
// RUN: %{ispc} %s --target=host --nowrap -O1 --emit-llvm-text --no-discard-value-names -o - | FileCheck %s -check-prefix=CHECKO1
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --no-discard-value-names -o - | FileCheck %s -check-prefix=CHECKO2


void compute(uniform float input[], uniform float output[], uniform int count) {
    foreach (i = 0 ... count) {
        output[i] = sin(input[i]) + (input[i]);
    }
}

// CHECKO1: Function Attrs: {{.*}} optsize
// CHECKO1-LABEL: @foo
// CHECKO1: @compute

// CHECKO2-NOT: Function Attrs: {{.*}} optsize
// CHECKO2-LABEL: @foo
// CHECKO2-NOT: @compute
unmasked void foo(uniform float input[], uniform float output[], uniform int count) { compute(input, output, count); }
