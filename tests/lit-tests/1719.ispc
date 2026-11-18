//; RUN: %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s
void test(uniform int64 var1, uniform int64 var2) {
    //; CHECK: Warning: Division by zero is undefined
    var1 = -25 / 0;
    //; CHECK: Warning: Remainder by zero is undefined
    var2 = -25 % 0;
}

void test_overflow(uniform int8 x, uniform int64 y) {
    //; CHECK: Warning: Binary expression with type "const uniform int8" can't represent value.
    x = (uniform int8)-128 / (uniform int8)-1;
    //; CHECK: Warning: Binary expression with type "const uniform int64" can't represent value.
    y = (uniform int64)0x8000000000000000ull % -1ll;
}
