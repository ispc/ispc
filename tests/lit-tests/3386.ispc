// This tests that variables declared in statements without braces are properly scoped
// RUN: not %{ispc} %s --nostdlib --nowrap --target=host -o %t.o 2>&1 | FileCheck %s

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
// CHECK: 29:20: Error: Undeclared symbol "z". Did you mean "x"?
// Test with else branch
int32 Test3(int32 x) {
    if (x == 0)
        varying int32 y = 2;
    else
        varying int32 z = 3;

    // Neither y nor z should be visible here
    return x + y + z;
}

// CHECK: 37:16: Error: Undeclared symbol "w". Did you mean "x"?
// Test with while loop - single statement without braces
int32 Test4(int32 x) {
    while (x < 5)
        varying int32 w = 4;
    return x + w;  // w should not be visible here
}

// CHECK: 46:16: Error: Undeclared symbol "v". Did you mean "x"?
// Test with do-while loop - single statement without braces
int32 Test5(int32 x) {
    do
        varying int32 v = 5;
    while (x < 10);
    return x + v;  // v should not be visible here
}
