//; RUN: %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s
void test(uniform int64 var1, uniform int64 var2) {
    //; CHECK: Warning: Division by zero is undefined
    var1 = -25 / 0;
    //; CHECK: Warning: Remainder by zero is undefined
    var2 = -25 % 0;
}
