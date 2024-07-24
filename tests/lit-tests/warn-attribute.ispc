// RUN: %{ispc} --target=host --nowrap --nostdlib  %s -o t.o 2>&1 | FileCheck %s

#define ATTR __attribute__((__unkNOWN_at))

// CHECK-COUNT-8: Warning: Ignoring unknown attribute "__unkNOWN_at".

ATTR void foo1();

ATTR void *func1();

void boo(ATTR int x);

ATTR uniform int x;
ATTR uniform int * uniform y;

void func() {
    ATTR int l1;
    ATTR uniform int l2;
    ATTR uniform int * uniform l3;
}
