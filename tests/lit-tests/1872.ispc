// RUN: not %{ispc} %s --target=host --nowrap > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// CHECK: Error:
// CHECK-NOT: Please file a bug
struct s {
    double d;
};
export uniform double foo(uniform s &x, uniform double y) { return x - y + y; }
