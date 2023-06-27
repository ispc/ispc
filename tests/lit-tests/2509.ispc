// RUN: not %{ispc} --target=avx1-i32x4 --nowrap %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process
struct S
{
    float x;
};

void f()
{
    S s = { b };
}
