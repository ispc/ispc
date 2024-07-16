// RUN: %{ispc} --target=host %s -o - --emit-llvm-text | FileCheck %s
// RUN: %{ispc} --target=host --debug-phase=2:2 %s -o t.o | FileCheck %s --check-prefix=CHECK-BAR

// CHECK: define <[[WIDTH:[0-9]+]] x i32> @foo(i32 %x, <[[WIDTH]] x i32> {{.*}} %y, <[[WIDTH]] x {{.*}}> %__mask)
__attribute__((unmangled)) int foo(uniform int x, int y) { return y; }

// CHECK-BAR: declare void @bar({{.*}})
__attribute__((unmangled)) void bar();

// CHECK-BAR: declare void @bar1()
__attribute__((unmangled)) unmasked void bar1();
