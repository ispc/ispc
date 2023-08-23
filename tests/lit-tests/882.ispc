// RUN: %{ispc} --target=host --nostdlib --nowrap -o - --emit-llvm-text %s 2>&1 | FileCheck %s

// CHECK-NOT: Error: Assertion failed
void main()
{
bool<3> a;
bool<3> b;
a && b;
}
