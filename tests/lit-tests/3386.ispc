// This tests that variables declared in if statements are properly scoped
// RUN: not %{ispc} %s --nostdlib --nowrap --target=host -o %t.o | FileCheck %s

// CHECK: 8:16: Error: Undeclared symbol "y". Did you mean "x"?
int32 Test1(int32 x) {
    if (x == 0)
        varying int32 y = 2;
    return x + y;  // y should not be visible here
}

// CHECK: 16:16: Error: Undeclared symbol "y". Did you mean "x"?
int32 Test2(int32 x) {
    if (x == 0) {
        varying int32 y = 2;
    }
    return x + y; // y should not be visible here
}

// CHECK: 29:16: Error: Undeclared symbol "y". Did you mean "x"?
// CHECK: 29:16: Error: Undeclared symbol "z". Did you mean "x"?
// Test with else branch
int32 Test3(int32 x) {
    if (x == 0)
        varying int32 y = 2;
    else
        varying int32 z = 3;

    // Neither y nor z should be visible here
    return x + y + z;
}
