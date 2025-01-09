// Check that there is no crash when using sizeof in global variables (#3084).
// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.ll --emit-llvm-text 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR:

uniform int test11 = sizeof(uniform int);
uniform int test12 = sizeof(varying int);
uniform size_t test21 = sizeof(uniform size_t);
uniform size_t test22 = sizeof(varying size_t);

varying int test31 = (varying int)sizeof(varying int);
varying int test32 = (varying int)sizeof(uniform int);
varying size_t test41 = (varying size_t)sizeof(varying size_t);
varying size_t test42 = (varying size_t)sizeof(uniform size_t);