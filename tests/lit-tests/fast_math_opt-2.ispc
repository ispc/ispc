// RUN: %{ispc} %s --target=host --opt=fast-math -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_IR
// RUN: %{ispc} %s --target=avx2-i32x8 --opt=fast-math -O2 --emit-asm -o -| FileCheck %s -check-prefix=CHECK_X64_ASM
// RUN: %{ispc} %s --target=avx2-i32x8 --opt=fast-math -O2 --emit-asm -o -| FileCheck %s -check-prefix=CHECK_X64_ASM_V2

// Must return the input
// CHECK_IR: @test1
// CHECK_IR: ret <8 x float> %v1
float test1(float v1) {
    return v1 + 1.0 - 1.0;
}

// Must return the input
// CHECK_IR: @test2
// CHECK_IR: ret <8 x float> %v2
float test2(float v2) {
    float t = v2 * 0.5;
    return t * 2;
}

// Must return the input
// CHECK_IR: @test3
// CHECK_IR: ret <8 x float> %v3
float test3(float v3) {
    float t = v3 * 0.5 + 2.5;
    return (t - 2.5) * 2.0;
}

// Must return the input
// CHECK_IR: @test4
// CHECK_IR: ret <8 x float> %v4
float test4(float v4) {
    float t = v4 * 0.5 + 2.5;
    return (t - 0.5) * 2.0 - 4.0;
}

// CHECK_X64_ASM_V2-COUNT-3: vfmadd{{[1-3][1-3][1-3]}}ps

// Must just generate a single FMA
// CHECK_X64_ASM: 0x40600000
// CHECK_X64_ASM: 0x40200000
// CHECK_X64_ASM: @test5
float test5(float v5) {
    return v5 * 3.5 + 2.5;
}

// Must just generate a single FMA
// CHECK_X64_ASM: 0xc1408000
// CHECK_X64_ASM: 0xc19f4000
// CHECK_X64_ASM: @test6
float test6(float v6) {
    v6 *= 2.5;
    v6 += 1.5;
    v6 *= 1.75;
    v6 += 3.25;
    v6 *= -2.75;
    v6 -= 3.75;
    return v6;
}

// Must just generate a single FMA (certainly only with the full fast-math)
// CHECK_X64_ASM: 0xc08f5076
// CHECK_X64_ASM: 0xc116f8b0
// CHECK_X64_ASM: @test7
float test7(float v7) {
    v7 *= 2.75;
    v7 += 1.25;
    v7 *= 1.5;
    v7 += 3.0;
    v7 /= -1.25;
    v7 -= 3.5;
    v7 += -0.0;
    v7 *= 9.5;
    v7 += 4.25;
    v7 /= 7;
    return v7;
}

// Must return 0 with an aggressive fast-math, but it must only generate a
// basic vsubps instruction due possible to NaN/Inf values in `v8`
// CHECK_X64_ASM: @test8
// CHECK_X64_ASM: vsubps
float test8(float v8) {
    return v8 - v8;
}
