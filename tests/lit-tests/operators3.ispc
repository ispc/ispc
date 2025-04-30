// RUN: not %{ispc} --target=host --nowrap %s -o - 2>&1 | FileCheck %s

struct MyStruct {
    int value;
};


noinline MyStruct operator-(MyStruct a, MyStruct b) {
    MyStruct result;
    result.value = a.value - b.value;
    return result;
}

// Partial implementation - only implements == for int comparison
bool operator==(MyStruct a, int b) {
    return a.value == b;
}

bool operator==(MyStruct a, MyStruct b) {
    return a.value == b.value;
}

// CHECK: 31:18: Error: operator operator+(varying struct MyStruct, varying struct MyStruct) is not defined.
// CHECK: 33:18: Error: Unable to find any matching overload for call to function "operator-".
// CHECK-NOT: Error
int test_missing_operator() {
    MyStruct a, b;
    a.value = 10;
    b.value = 20;

    MyStruct c = a + b;

    MyStruct d = a - 4;

    if (c == d) {
        return 0;
    }
    if (c == 4) {
        return 1;
    }
    return 2;
}