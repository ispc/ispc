// RUN: not %{ispc} --target=host --nowrap --nostdlib  %s -o t.o 2>&1 | FileCheck %s

// CHECK: Error: "noescape" attribute illegal with "varying" parameter "ptr".
void foo(__attribute__((noescape)) uniform int *ptr);

// CHECK: Error: "noescape" attribute illegal with non-pointer parameter "x".
void func(__attribute__((noescape)) uniform int x);

// CHECK: Warning: Ignoring "noescape" attribute for function "func_1".
__attribute__((noescape)) void *func_1();

// CHECK: Warning: Ignoring "noescape" attribute for function "func_2".
__attribute__((noescape)) void func_2();

// It is not yet supported to have __attribute__ in struct declaration.
// CHECK: Error: syntax error, unexpected '__attribute__'
struct S {
  __attribute__((noescape)) void *ptr_str;
};
