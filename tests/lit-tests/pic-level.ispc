// RUN: %{ispc} --target=host --target-os=linux --nostdlib --nowrap %s -o - --emit-llvm-text 2>&1 | FileCheck %s --check-prefix=CHECK-NOPIC
// RUN: %{ispc} --target=host --target-os=linux --nostdlib --nowrap %s -o - --emit-llvm-text --pic 2>&1 | FileCheck %s --check-prefix=CHECK-PIC
// RUN: %{ispc} --target=host --target-os=linux --nostdlib --nowrap %s -o - --emit-llvm-text --PIC 2>&1 | FileCheck %s --check-prefix=CHECK-BIGPIC

// Note: in LLVM 16 the merge behaviour of PIC flags for Modules changed.
// Before, the merge value was the maximal. We are merging the flags from the
// user module, the builtin-cpp module and the target module. So, the PIC level
// of the resulted code depends not only on the user flags, but also on the
// flags that we build builtin and target modules.
// REQUIRES: LLVM_16_0+ && LINUX_ENABLED


// TODO: see tests/lit-tests/pic-level-1.ispc for more details.
// CHECK-NOPIC: !"PIC Level", i32 2

// CHECK-PIC: !"PIC Level", i32 1
// CHECK-BIGPIC: !"PIC Level", i32 2

uniform int global_var = 42;

uniform int foo() { return global_var; }
