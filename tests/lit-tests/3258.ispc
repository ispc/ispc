// This test checks that the ISPC compiler can handle the following constructs without crashing.
// The test is not expected to produce any output.
// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR
// CHECK: 9:30: Error: Array initializer must be an initializer list

export uniform float test(const uniform float values[][3]) {
    uniform float value[3] = values[programIndex];
    return value[0];
}
