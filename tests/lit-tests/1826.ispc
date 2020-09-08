//; RUN: %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s
//; CHECK-NOT: 5:35: Warning: Binary expression with type "const uniform int32" can't represent value.
//; CHECK: 6:33: Warning: Binary expression with type "const uniform int32" can't represent value.
void foo() {
    uniform unsigned int nowarn = 1 << 31;
    uniform unsigned int warn = 1 << 32;
}
