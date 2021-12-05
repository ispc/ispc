//; RUN: not %{ispc} %s --target=host -O0 --nowrap 2>&1 | FileCheck %s

// Differences with C++.

// CHECK: 7:15: Error: syntax error
// Hexadecimal FP literals may start only from 0 and 1.
float f1 = 0x2p1;

// Common with C++.
// CHECK: 12:12: Error: syntax error
// There must be at least one digit before or after the period.
float f2 = .f;

// Fixed bugs in literals definition
// "f" and "f16" suffix should not appear in the FP literal in scientific form twice.
// These constants should not be parsed and an error should be reported.
// CHECK: 18:16: Error: syntax error
float f3 = 0.1fe+1f;
// CHECK: 20:18: Error: syntax error
float f4 = 0.1f16e+1f16;
