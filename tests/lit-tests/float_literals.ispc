//; RUN: %{ispc} %s --target=host --emit-asm -O0 --nostdlib -o - | FileCheck %s

// Lexing and parsing focused test for double floating point literals.

// no exponent
// CHECK: d2:
// CHECK: 0x4000000000000000
// CHECK: 0x3fe0000000000000
// CHECK: 0x4004000000000000
// uniform double d1 = 2d; // this is illegal
uniform double d2 = 2.d;
uniform double d3 = .5d;
uniform double d4 = 2.5d;

// exponent without sign
// CHECK: d5:
// CHECK: 0x4034000000000000
// CHECK: 0x4034000000000000
// CHECK: 0x4014000000000000
// CHECK: 0x4039000000000000
uniform double d5 = 2e1d;
uniform double d6 = 2.e1d;
uniform double d7 = .5e1d;
uniform double d8 = 2.5e1d;

// exponent with + sign
// CHECK: d9:
// CHECK: 0x4034000000000000
// CHECK: 0x4034000000000000
// CHECK: 0x4014000000000000
// CHECK: 0x4039000000000000
uniform double d9 = 2e+1d;
uniform double d10 = 2.e+1d;
uniform double d11 = .5e+1d;
uniform double d12 = 2.5e+1d;

// exponent with - sign
// CHECK: d13:
// CHECK: 0x3fc999999999999a
// CHECK: 0x3fc999999999999a
// CHECK: 0x3fa999999999999a
// CHECK: 0x3fd0000000000000
uniform double d13 = 2e-1d;
uniform double d14 = 2.e-1d;
uniform double d15 = .5e-1d;
uniform double d16 = 2.5e-1d;

// Fortran double format
// CHECK: d17:
// CHECK: 0x4034000000000000
// CHECK: 0x4034000000000000
// CHECK: 0x4014000000000000
// CHECK: 0x4039000000000000
uniform double d17 = 2d1;
uniform double d18 = 2.d1;
uniform double d19 = .5d1;
uniform double d20 = 2.5d1;
