// RUN: %{ispc} --target=host --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// CHECK-NOT: Error: Multiple types provided for declaration

typedef struct Foo {
    int a;
} Foo;

// CHECK: {{.*}} = type { <{{.*}} x i32> }
// CHECK: @global_foo = {{.*}} global {{.*}}varying_Foo zeroinitializer
Foo global_foo;
