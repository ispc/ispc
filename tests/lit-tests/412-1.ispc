// This test checks that we can define a pointer to function type.

// RUN: %{ispc} --nowrap --nostdlib --target=host --debug-phase=10:10 -o %t.o %s 2>&1 | FileCheck %s

// CHECK: @b = global ptr @foo2___uni
// CHECK-NOT: FATAL ERROR:

uniform int foo2(uniform int b);

typedef uniform int (intFct)(uniform int);

intFct *uniform b = foo2;

