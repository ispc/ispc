// Case 1: Checks avx512spr-x16 assembly output for zmm registers.
//; RUN: %{ispc} %s --target=avx512spr-x16 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_ZMM_32x16
// Case 2a: Checks avx512spr-x4 assembly output to ensure there are no zmm registers.
//; RUN: %{ispc} %s --target=avx512spr-x4 --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"
// Case 2b: Checks avx512spr-x8 assembly output to ensure there are no zmm registers.
//; RUN: %{ispc} %s --target=avx512spr-x8 --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"
// Case 3: Checks avx512spr-x16 LLVM IR to ensure avx512spr-x8 func attributes are not present.
//; RUN: %{ispc} %s --target=avx512spr-x16 --emit-llvm -o %t_8.bc
//; RUN: llvm-dis %t_8.bc -o - | FileCheck %s --implicit-check-not ""min-legal-vector-width"="256"" --implicit-check-not ""prefer-vector-width"="256""
// Case 4a: Checks avx512spr-x4 LLVM IR to ensure avx512spr-x4 func attributes are present (same as for x8).
//; RUN: %{ispc} %s --target=avx512spr-x4 --emit-llvm -o %t_4.bc
//; RUN: llvm-dis %t_4.bc -o - | FileCheck %s -check-prefix=CHECK_ATTR_8
// Case 4b: Checks avx512spr-x8 LLVM IR to ensure avx512spr-x8 func attributes are present.
//; RUN: %{ispc} %s --target=avx512spr-x8 --emit-llvm -o %t_8.bc
//; RUN: llvm-dis %t_8.bc -o - | FileCheck %s -check-prefix=CHECK_ATTR_8

// REQUIRES: X86_ENABLED && LLVM_14_0+ && !MACOS_HOST

double ret_round(double val)
{
//; CHECK_ZMM_32x16: zmm
    return round(val+.49999);
}

double foo(double val1, double val2)
{
//; CHECK_ZMM_32x16: zmm
    double sub_val = val1 - val2;
    return ret_round(sub_val);
}

export void infoo(uniform double afoo1[], uniform double afoo2[], uniform double afoo3[])
{
    double a = afoo1[programIndex];
    double b = afoo2[programIndex];
    double c = foo(a, b);
//; CHECK_ZMM_32x16: zmm
    afoo3[programIndex] = c;
}
// The avx512spr-x8 specific function attribute list will be present at 2 places.
// Once where function attributes for exported functions are aggregated and
// again where function attributes for non-exported functions are aggregated.
//; CHECK_ATTR_8: "min-legal-vector-width"="256"
//; CHECK_ATTR_8: "prefer-vector-width"="256"
//; CHECK_ATTR_8: "min-legal-vector-width"="256"
//; CHECK_ATTR_8: "prefer-vector-width"="256"
