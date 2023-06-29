// Check that ISPC produces error for incorrect code and does not crash. ISPC issue #2341.
// RUN: not %{ispc} %s --nowrap --nostdlib --target=host 2>&1 | FileCheck %s

// CHECK: Error: Can't convert from type "const varying int32" to type "uniform unsigned int16" for initializer
// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process
void test_func(uniform uint16 start_index) {
    uniform uint16 step = 4 * programIndex;
    uniform uint16 vertex_id = start_index;
    vertex_id += step;
}

