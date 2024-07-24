// RUN: not %{ispc} --target=host --nowrap --nostdlib  %s -o t.o 2>&1 | FileCheck %s

// At the moment, we don't support __attribute__ for struct and its members. It
// can be supported in the future if needed.

// CHECK: Error: syntax error, unexpected '__attribute__', expecting {{.*}}
struct __attribute__((__packed__)) S {
    int b;
};

// CHECK: Error: syntax error, unexpected '__attribute__'.
struct S1 {
    __attribute__((aligned(8))) int a;
};

// CHECK: Error: syntax error, unexpected '-', expecting identifier.
__attribute__((-1)) int a;

// CHECK: Error: syntax error, unexpected ')'
__attribute__((aligned())) int a;
