// RUN: not %{ispc} --target=host --nowrap %s -o - 2>&1 | FileCheck %s

struct MyStruct {
    int value;
};

struct AnotherStruct {
    float value;
};

struct ComplexStruct {
    MyStruct s1;
    AnotherStruct s2;
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

// Define compound assignment operator
noinline MyStruct operator+=(MyStruct &a, MyStruct b) {
    a.value += b.value;
    return a;
}

// Incorrect signature - parameter should be a reference for compound assignment
noinline AnotherStruct operator-=(AnotherStruct a, AnotherStruct b) {
    a.value -= b.value;
    return a;
}

// Define unary operator - correct
noinline MyStruct operator++(MyStruct &a) {
    a.value++;
    return a;
}

// Define postfix operator - incorrect (missing int parameter)
noinline AnotherStruct operator++(AnotherStruct &a) {
    AnotherStruct temp = a;
    a.value += 1.0f;
    return temp;
}

// Define postfix operator correctly (with int parameter)
noinline AnotherStruct operator++(AnotherStruct &a, int) {
    AnotherStruct temp = a;
    a.value += 1.0f;
    return temp;
}

// CHECK: 80:18: Error: operator operator+(varying struct MyStruct, varying struct MyStruct) is not defined.
// CHECK: 80:18: Error: First operand to binary operator "+" is of invalid type "varying struct MyStruct".
// CHECK: 81:18: Error: Unable to find any matching overload for call to function "operator-".
// CHECK: 81:18: Error: First operand to binary operator "-" is of invalid type "varying struct MyStruct".
// CHECK: 105:20: Error: Unable to find any matching overload for call to function "operator-".
// CHECK: 105:21: Error: Negate not allowed for non-numeric type "varying struct MyStruct"
// CHECK: 108:25: Error: operator operator+(varying struct MyStruct, varying struct AnotherStruct) is not defined.
// CHECK: 108:25: Error: First operand to binary operator "+" is of invalid type "varying struct MyStruct".
// CHECK: 118:18: Error: Unable to find any matching overload for call to function "operator==".
// CHECK: 118:18: Error: First operand to operator "==" is of non-comparable type "varying struct ComplexStruct".
// CHECK: 115:26: Error: Can't convert between different struct types "varying struct ComplexStruct" and "varying struct MyStruct" for initializer.
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

void test_additional_errors() {
    MyStruct ms1, ms2;
    ms1.value = 5;
    ms2.value = 10;

    AnotherStruct as1, as2;
    as1.value = 2.5f;
    as2.value = 1.5f;

    ms1 += ms2;  // This should work fine

    // Missing unary minus operator
    MyStruct neg = -ms1;

    // Type mismatch in operator usage
    MyStruct combined = ms1 + as1;  // Type mismatch

    ComplexStruct cs;
    cs.s1.value = 42;
    cs.s2.value = 3.14f;

    // Implicit conversion not available for struct types
    MyStruct converted = cs;

    // Using unrelated struct type with comparison operator
    bool comp = (cs == ms1);
}
