// RUN: %{ispc} %s --target=avx2 -O2 --opt=fast-math  -h %t.h --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX
// RUN: %{ispc} %s --target=avx1 -O2 --opt=fast-math  -h %t.h --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX
// RUN: %{ispc} %s --target=sse4 -O2 --opt=fast-math  -h %t.h --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --target=sse2 -O2 --opt=fast-math  -h %t.h --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SSE
// REQUIRES: X86_ENABLED && LLVM_14_0+
export void test_interleaved(const uniform float xin[],
                             const uniform float yin[],
                             const uniform float zin[],
                             uniform int data[]) {
    foreach (i = 0 ... 256) {
// CHECK_AVX: vunpckhps
// CHECK_AVX: vunpcklps
// CHECK_AVX-NOT: vpextrd

// CHECK_SSE: unpckhps
// CHECK_SSE: unpcklps
// CHECK_SSE-NOT: vpextrd
        int data0 = (2 * xin[i] + 3 * yin[i] + 2 * zin[i]) / 7.0f;
        int data1 = (xin[i] + yin[i] + zin[i]) / 3.0f;
        *(data + 2 * i + 0) = data0;
        *(data + 2 * i + 1) = data1;
    }
}
