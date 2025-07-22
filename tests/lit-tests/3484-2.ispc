// This test ensures that the compiler doesn't crash with a segmentation fault
// when trying to use an invalid function call in an initializer list.
// Previously this would cause a null pointer dereference in HasAtomicInitializerList.

// RUN: not %{ispc} %s --target=host --nowrap 2>&1 | FileCheck %s
// CHECK: Error: Must provide function name or function pointer for function call expression
export void func(uniform float values[][programCount]) {
    uniform float<3> col_xyz = {values(&values[0][0])};
}