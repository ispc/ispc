// RUN: %{ispc} --target=host --nostdlib --nowrap %s -o - --emit-llvm-text 2>&1 | FileCheck %s
// RUN: %{ispc} --target=host --nostdlib --nowrap %s -o - --emit-llvm-text --mcmodel=small 2>&1 | FileCheck %s
// RUN: %{ispc} --target=host --nostdlib --nowrap %s -o - --emit-llvm-text --mcmodel=large 2>&1 | FileCheck %s

// XFAIL: *

// TODO:
// There is a strange behavior in the current PIC handling. When --pic
// omitted LLVM IR contains "PIC Level", i32 2 but TargetMachine object is not
// initialized with relocModel = PIC_. It looks wrong and should be fixed but
// preserve this behavior for now.
// This especially strange considering that --mcmodel=large|small affects the
// PIC Level when --pic is omitted.

// CHECK: !"PIC Level", i32 0
uniform int global_var = 42;

uniform int foo() { return global_var; }
