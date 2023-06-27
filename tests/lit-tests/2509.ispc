// RUN: not %{ispc} --target=host --nowrap --nostdlib %s 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process
struct S
{
    float x;
};

void f()
{
    S s = { b };
}
