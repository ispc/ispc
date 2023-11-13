// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Left hand side of assignment expression can't be assigned to

int foo() {return 2;}

int bar()
{
    foo() = 2;
}
