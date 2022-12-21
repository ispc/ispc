//; RUN: %{ispc} %s --target=avx2-i32x4    --emit-asm -O0 --nostdlib -o - | FileCheck %s
//; RUN: %{ispc} %s --target=avx512knl-x16 --emit-asm -O0 --nostdlib -o - | FileCheck %s
//; RUN: %{ispc} %s --target=avx512skx-x4  --emit-asm -O0 --nostdlib -o - | FileCheck %s
//; RUN: %{ispc} %s --target=neon-i32x4    --emit-asm -O0 --nostdlib -o - | FileCheck %s

//; RUN: not %{ispc} %s --target=sse2-i32x4 --emit-asm -O0 --nostdlib
//; RUN: not %{ispc} %s --target=sse4-i32x4 --emit-asm -O0 --nostdlib
//; RUN: not %{ispc} %s --target=avx1-i32x4 --emit-asm -O0 --nostdlib

// REQUIRES: X86_ENABLED

// Formats (sign-exponent-matissa)
// float16: 1-5-10  / exponent range [-14, 15]
// float:   1-8-23  / exponent range [-126, 127]
// double:  1-11-52 / exponent range [-1022, 1023]

// Biased exponent (w bits):
// - integers between 1 and 2^w-2 - normal numbers;
// - 0 is reserved for +/-0 and subnormals;
// - 2^w-1 is reserved for +/-inf and NaNs;

// Infinity is represented by the largest biased exponent allowed by the format and a mantissa of zero.

// Test 3 types of literals:
// - decimal
// - scientific
// - hexadecimal
// for 3 floating point types:
// - float16
// - float
// - double

// Also, as an exception from general rule "Fortran double" format is accepted:
// - scientific format with "d" or "D" used instead of "e" is treated as double.

// Use f/F, f16/F16, d/D, e/E, p/P, and 0x/0X  in different combinations to cover more cases in lexer.

// A testing methodology remark: we check assembler output, as it prints values
// in binary representation, which is easy to verify.
// While LLVM IR prints floating point values with random beautification - i.e.
// depending on the value it may print scientific or binary form. In the binary
// form it prints "double" representation for both "float" and "double".


////////////////////////////////////////////////////////////////////////////////
// float16
////////////////////////////////////////////////////////////////////////////////

// smallest positive subnormal number (0-00000-0000000001)
// CHECK-LABEL: h1_v1:
// CHECK-COUNT-3: 0x0001
uniform float16 h1_v1 = 0.000000059604645f16;
uniform float16 h1_v2 = 5.9604645e-8f16;
uniform float16 h1_v3 = 0x1p-24f16;
// largest positive subnormal number  (0-00000-1111111111)
// CHECK-LABEL: h2_v1:
// CHECK-COUNT-3: 0x03ff
uniform float16 h2_v1 = 0.000060975552F16;
uniform float16 h2_v2 = 6.0975552e-5F16;
uniform float16 h2_v3 = 0x0.ffcp-14F16;
// smallest positive normal number    (0-00001-0000000000)
// CHECK-LABEL: h3_v1:
// CHECK-COUNT-3: 0x0400
uniform float16 h3_v1 = 0.00006103515625f16;
uniform float16 h3_v2 = 6.103515625E-5f16;
uniform float16 h3_v3 = 0x1P-14f16;
// largest positive normal number     (0-11110-1111111111)
// CHECK-LABEL: h4_v1:
// CHECK-COUNT-3: 0x7bff
uniform float16 h4_v1 = 65504.F16;
uniform float16 h4_v2 = 6.5504e4F16;
uniform float16 h4_v3 = 0X1.ffcp+15F16;
// nearest value to 1/3
// CHECK-LABEL: h5_v1:
// CHECK-COUNT-3: 0x3555
uniform float16 h5_v1 = 0.33325195f16;
uniform float16 h5_v2 = 0.33325195e+0f16;
uniform float16 h5_v3 = 0x1.554p-2f16;
// largest number less than one
// CHECK-LABEL: h6_v1:
// CHECK-COUNT-3: 0x3bff
uniform float16 h6_v1 = 0.99951172F16;
uniform float16 h6_v2 = 0.99951172E-0F16;
uniform float16 h6_v3 = 0x1.ffcP-1F16;
// one
// CHECK-LABEL: h7_v1:
// CHECK-COUNT-3: 0x3c00
uniform float16 h7_v1 = 1.0f16;
uniform float16 h7_v2 = 1.e+0f16;
uniform float16 h7_v3 = 0x1p+0f16;
// smallest number larger than one
// CHECK-LABEL: h8_v1:
// CHECK-COUNT-3: 0x3c01
uniform float16 h8_v1 = 1.00097656F16;
uniform float16 h8_v2 = 1.00097656e0F16;
uniform float16 h8_v3 = 0X1.004p0F16;
// +0
// CHECK-LABEL: h9_v1
// CHECK-COUNT-3: {{zerofill|0x0000}}
uniform float16 h9_v1 = 0.0f16;
uniform float16 h9_v2 = 0.0E+0f16;
uniform float16 h9_v3 = 0x0P0f16;
// -0
// CHECK-LABEL: h10_v1:
// CHECK-COUNT-3: 0x8000
uniform float16 h10_v1 = -0.0F16;
uniform float16 h10_v2 = -0.0e+0F16;
uniform float16 h10_v3 = -0x0p0F16;
// +Inf
// 65520 is the first to round to Inf
// 1.e+5 is some value larger than maximum representable.
// 0x1p16 is exact bit representation of Inf
// CHECK-LABEL: h11_v1:
// CHECK-COUNT-3: 0x7c00
uniform float16 h11_v1 = 65520.f16;
uniform float16 h11_v2 = 1.e+5f16;
uniform float16 h11_v3 = 0x1p16f16;
// -Inf
// CHECK-LABEL: h12_v1:
// CHECK-COUNT-3: 0xfc00
uniform float16 h12_v1 = -65520.F16;
uniform float16 h12_v2 = -1.E+5F16;
uniform float16 h12_v3 = -0X1P16F16;
