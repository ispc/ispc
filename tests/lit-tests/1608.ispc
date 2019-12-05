//; RUN: %{ispc} %s --target=avx512knl-i32x16 --emit-asm -o - | FileCheck %s --implicit-check-not "vpcmpeqb" -check-prefix=CHECK_AVX512knl16
//; RUN: %{ispc} %s --target=avx512skx-i32x16 --emit-asm -o - | FileCheck %s --implicit-check-not "vpcmpeqb" -check-prefix=CHECK_AVX512skx16
//; RUN: %{ispc} %s --target=avx512skx-i32x8 --emit-asm -o - | FileCheck %s --implicit-check-not "vpcmpeqw" --implicit-check-not "vpcmpneqw" -check-prefix=CHECK_AVX512skx8
// REQUIRES: LLVM_8_0+
void int_compare(uniform float vin[], uniform int cmp[], uniform float vout[], uniform int count) {
    foreach (index = 0 ... count) {
        varying float v = vin[index];
        varying int c = cmp[index];
        if (c < 0)
            v = v * v;
        else
            v = sqrt(v);

        vout[index] = v;
    }
}
