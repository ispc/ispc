// This test checks that the ISPC compiler can handle the following constructs without crashing.
// The test is not expected to produce any output.
// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR
// CHECK: 15:15: Error: Array type "varying float[4]" can't be converted to unsized array "varying float[]" for type cast expression.
// CHECK: 10:5: Error: Illegal to declare an unsized array variable without providing an initializer expression to set its size.

void test1(float RET[]) {
    float values[];
    RET = values;
}
void test2() {
    float values[4];
    (float [])values;
}
