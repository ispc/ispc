// This test checks generated code for the integer division in case when divisor is a constant

// RUN: %{ispc} -O2 --target=host %s --emit-asm --x86-asm-syntax=intel -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// Check that we don't generate a chain of scalar imul instructions
// CHECK-NOT: imul
// int foo5(int x) { return x / 5; }

// CHECK-LABEL: foo0___vyi:
// CHECK-NEXT: bb
// CHECK-NEXT: ret
int foo0(int x) { return x / 0; }
