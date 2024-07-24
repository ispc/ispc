// RUN: %{ispc} --target=host --nowrap --nostdlib  %s -o t.o 2>&1 | FileCheck %s

#define ATTR __attribute__((__unkNOWN_at))

// CHECK-COUNT-5: Warning: Ignoring unknown attribute "__unkNOWN_at".

ATTR void foo1() { }

ATTR void *func1() { }

void boo(ATTR uniform int* x);

void boo1(ATTR uniform int* x) { }

void boo2(ATTR int x) { }
