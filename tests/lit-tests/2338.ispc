// This test checks that the compiler reports errors when mixing internal
// qualifiers (e.g. `inline` and `static`) with external ones (e.g. `export`
// `extern "C"` and `extern "SYCL"`).

// RUN: not %{ispc} --target=host --emit-llvm-text --nostdlib --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Function qualifier 'export' incompatible with 'inline'
// CHECK: Error: Function qualifier 'extern' incompatible with 'inline'
// CHECK: Error: Function qualifier 'extern' incompatible with 'inline'
// CHECK: Error: Function qualifier 'export' incompatible with 'static'
// CHECK-COUNT-2: Error: Multiple storage class specifiers in a declaration are illegal
// CHECK-NOT: FATAL ERROR:

inline  export         void test1() { }
inline  extern "C"     void test2() { }
inline  extern "SYCL"  void test3() { }
static  export         void test4() { }
static  extern "C"     void test5() { }
static  extern "SYCL"  void test6() { }
