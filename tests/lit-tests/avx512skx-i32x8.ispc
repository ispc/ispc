// Case 1: Checks avx512skx-i32x16 assembly output for zmm registers.
//; RUN: %{ispc} %s --target=avx512skx-i32x16 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_ZMM_32x16
// Case 2: Checks avx512skx-i32x8 assembly output to ensure there are no zmm registers.
//; RUN: %{ispc} %s --target=avx512skx-i32x8 --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"
// Case 3: Checks avx512skx-i32x16 LLVM IR to ensure avx512skx-i32x8 func attributes are not present.
//; RUN: %{ispc} %s --target=avx512skx-i32x16 --emit-llvm -o %t_8.bc
//; RUN: llvm-dis %t_8.bc -o - | FileCheck %s --implicit-check-not ""min-legal-vector-width"="256"" --implicit-check-not ""prefer-vector-width"="256""
// Case 4: Checks avx512skx-i32x8 LLVM IR to ensure avx512skx-i32x8 func attributes are present.
//; RUN: %{ispc} %s --target=avx512skx-i32x8 --emit-llvm -o %t_8.bc
//; RUN: llvm-dis %t_8.bc -o - | FileCheck %s -check-prefix=CHECK_ATTR_8

// REQUIRES: X86_ENABLED

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
// The avx512skx-i32x8 specific function attribute list will be present at 2 places.
// Once where function attributes for exported functions are aggregated and
// again where function attributes for non-exported functions are aggregated.
//; CHECK_ATTR_8: "min-legal-vector-width"="256"
//; CHECK_ATTR_8: "prefer-vector-width"="256"
//; CHECK_ATTR_8: "min-legal-vector-width"="256"
//; CHECK_ATTR_8: "prefer-vector-width"="256"
