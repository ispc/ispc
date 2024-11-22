// RUN: not %{ispc} %s --nowrap --target=host --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s

#define unmangled __attribute__((unmangled))
#define ATTRS unmasked unmangled

// CHECK: Error: LLVM intrinsic "llvm.ispc.unknown" not supported.

ATTRS void foo() {
    return @llvm.ispc.unknown();
}
