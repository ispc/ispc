// RUN: not %{ispc} --nostdlib --nowrap --target=host %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process
// CHECK: has no member named "f"
struct D {
    int h;
};
struct D d;
int foo() {
     return d.f;
}
