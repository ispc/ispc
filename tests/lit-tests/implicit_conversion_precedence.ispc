// RUN: %{ispc} %s --nowrap --target=host -o %t.o 2>&1 | FileCheck %s

// Test that implicit conversion warnings are issued when mixing float with int64/uint64

void test_float_uint64() {
    float f = 0.1f;
    uint64 u64 = 1;

    // CHECK: Warning: Implicit conversion between "uniform float" and "uniform uint64" for binary arithmetic operation. ISPC converts to "uniform uint64", which may differ from C/C++ behavior. Use explicit cast to avoid ambiguity.
    float result = f * u64;
}

void test_float_int64() {
    float f = 0.1f;
    int64 i64 = 1;

    // CHECK: Warning: Implicit conversion between "uniform float" and "uniform int64" for binary arithmetic operation. ISPC converts to "uniform int64", which may differ from C/C++ behavior. Use explicit cast to avoid ambiguity.
    float result = f * i64;
}

void test_double_uint64() {
    double d = 0.1;
    uint64 u64 = 1;

    // CHECK: Warning: Implicit conversion between "uniform double" and "uniform uint64" for binary arithmetic operation. ISPC converts to "uniform double", which may differ from C/C++ behavior. Use explicit cast to avoid ambiguity.
    double result = d * u64;
}

void test_double_int64() {
    double d = 0.1;
    int64 i64 = 1;

    // CHECK: Warning: Implicit conversion between "uniform double" and "uniform int64" for binary arithmetic operation. ISPC converts to "uniform double", which may differ from C/C++ behavior. Use explicit cast to avoid ambiguity.
    double result = d * i64;
}

// Test that no warning is issued for float with int32/uint32
void test_float_uint32() {
    float f = 0.1f;
    uint u32 = 1;

    // No warning expected for uint32
    float result = f * u32;
}

void test_float_int32() {
    float f = 0.1f;
    int i32 = 1;

    // No warning expected for int32
    float result = f * i32;
}
