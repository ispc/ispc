// This test checks that avx512skx-x16:
// - uses only zmm registers by default (for 512 bit vectors)
// - uses only ymm registers when --opt=disable-zmm is passed.
//
// This code is especially problematic, as shuffle generates a sequence of gep
// instructions, which are later combined to shuffle by code gen.

// RUN: %{ispc} %s --nowrap --emit-asm --target=avx512skx-x16 -o - 2>&1 | FileCheck %s -check-prefix=ZMM
// RUN: %{ispc} %s --nowrap --emit-asm --target=avx512skx-x16 --opt=disable-zmm -o - 2>&1 | FileCheck %s -check-prefix=YMM

// REQUIRES: X86_ENABLED

// ZMM-NOT: ymm
// YMM-NOT: zmm

// YMM: Warning: Using --opt=disable-zmm for avx512skx-x16 target can lead to compiler crashes, consider using the generic target instead: `--target=generic-i1x16 --cpu=skx --opt=disable-zmm`

struct FM {
    float M[16];
};

unmasked inline uniform FM bar(const uniform FM &m) {
    uniform FM r;

    uniform float *uniform mp = (uniform float *uniform) & m;
    uniform float *uniform rp = (uniform float *uniform) & r;

    static const varying int perm = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

    const varying float s = mp[programIndex];
    rp[programIndex] = shuffle(s, perm);

    return r;
}

export void foo(uniform FM out[], const uniform FM src[], const uniform int n) {
    for (uniform unsigned int i = 0; i < n; i++) {
        out[i] = bar(src[i]);
    }
}
