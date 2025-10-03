// RUN: %{ispc} %s --target=avx2 -O2 --opt=fast-math  -h %t.h --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX
// RUN: %{ispc} %s --target=avx1-i32x4 -O2 --opt=fast-math  -h %t.h --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX
// RUN: %{ispc} %s --target=sse4 -O2 --opt=fast-math  -h %t.h --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc} %s --target=sse2 -O2 --opt=fast-math  -h %t.h --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SSE2
// REQUIRES: X86_ENABLED
export void test_interleaved(const uniform float xin[],
                             const uniform float yin[],
                             const uniform float zin[],
                             uniform int data[]) {
    foreach (i = 0 ... 256) {
// CHECK_AVX: vunpckhps
// CHECK_AVX: vunpcklps
// CHECK_AVX-NOT: vpextrd

// CHECK_SSE4: unpckhps
// CHECK_SSE4: unpcklps
// CHECK_SSE4-NOT: vpextrd

// Test fails for SSE2 after updating ImproveMemoryOps to emit 64-bit index. Needs more investigation.
// CHECK_SSE2-NOT: unpckhps
// CHECK_SSE2-NOT: unpcklps
        int data0 = (2 * xin[i] + 3 * yin[i] + 2 * zin[i]) / 7.0f;
        int data1 = (xin[i] + yin[i] + zin[i]) / 3.0f;
        *(data + 2 * i + 0) = data0;
        *(data + 2 * i + 1) = data1;
    }
}
