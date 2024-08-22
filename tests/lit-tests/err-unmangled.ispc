// RUN: not %{ispc} --nowrap --target=host %s -o - --emit-llvm-text 2>&1 | FileCheck %s

#define UNMANGLED __attribute__((unmangled))

CHECK: Error: Function can't have both "extern" "C" and "unmangled" qualifiers
UNMANGLED extern "C" void bar();

// CHECK: Error: Function can't have both "extern" "SYCL" and "unmangled" qualifiers
UNMANGLED extern "SYCL" void func();

// CHECK: Error: Function can't have both "export" and "unmangled" qualifiers
UNMANGLED export void foo();
