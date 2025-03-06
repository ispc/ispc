// RUN: %{ispc} --nowrap --nostdlib --target=host --debug-phase=10:10 -o t.o -O2 %s 2>&1 | FileCheck %s

typedef export uniform int (*uniform intFct)(uniform int);

export uniform int foo(uniform int b) { return b; }
// CHECK: @f_ptr_gen = global ptr @foo___uni
intFct f_ptr_gen = foo;

export __attribute__((external_only)) uniform int bar(uniform int b) { return b; }
// CHECK: @f_ptr_ex = global ptr @bar
// CHECK-NOT: @f_ptr_ex = {{.*}} global ptr @bar___uni
intFct f_ptr_ex = bar;

// CHECK: call i32 @bar(
// CHECK: call i32 @foo___uni(
uniform int func(uniform int x) {
    return bar(x) + foo(x);
}

// CHECK-NOT: FATAL ERROR:
