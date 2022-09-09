//; RUN: not %{ispc} %s --target=host -O0 --nowrap 2>&1 | FileCheck %s

////////////////////////////////////////////////////////////////////////////////
// Differences with C++.

// Hexadecimal FP literals may start only from 0 and 1.
// CHECK: 8:15: Error: syntax error
float f1 = 0x2p1;

////////////////////////////////////////////////////////////////////////////////
// Common with C++.
// There must be at least one digit before or after the period.
// CHECK: 14:12: Error: syntax error
float f2 = .f;

// FP literal need to have a radix separator, even if a suffix is used (float16 or double).
// It's legal to have a constant without radix separator with "f" suffix (float) - deprecated.
// CHECK: 19:14: Error: floating point literal should have a radix separator
float16 f3 = 1f16;
// CHECK: 21:12: Warning: single precision floating point literal should have a radix separator
float f4 = 1f;
// CHECK: 23:13: Error: floating point literal should have a radix separator
double f5 = 1d;

////////////////////////////////////////////////////////////////////////////////
// Fixed bugs in literals definition
// "f" and "f16" suffix should not appear in the FP literal in scientific form twice.
// These constants should not be parsed and an error should be reported.
// CHECK: 30:16: Error: syntax error
float f6 = 0.1fe+1f;
// CHECK: 32:18: Error: syntax error
float f7 = 0.1f16e+1f16;
