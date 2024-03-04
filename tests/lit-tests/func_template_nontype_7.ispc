// This test is checking that template non-type parameters
// can be used to specify array dimension.
// RUN: %{ispc} %s --target=host --nostdlib 2>&1 | FileCheck %s
// CHECK-NOT: Error
// XFAIL: *
template<int Rows, int Cols>
noinline void foo(int matrix[Rows][Cols]) {
    // Process matrix
}

int bar() {
    int myMatrix[3][3];
    foo<3, 3>(myMatrix);
}