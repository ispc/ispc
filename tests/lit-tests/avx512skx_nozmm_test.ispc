// Test that --opt=disable-zmm or -nozmm works correctly with avx512skx-x16 target and
// that the generated code uses ymm registers only and the function attributes are set
// correctly in the emitted LLVM IR.
// RUN: %{ispc} %s --target=avx512skx-x16 --opt=disable-zmm --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_256
// RUN: %{ispc} %s --target=avx512skx-x16 --opt=disable-zmm --emit-asm -o - | FileCheck %s --implicit-check-not "%zmm"
// RUN: %{ispc} %s --target=avx512skx-x16-nozmm --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_256
// RUN: %{ispc} %s --target=avx512skx-x16-nozmm --emit-asm -o - | FileCheck %s --implicit-check-not "%zmm"

// CHECK_ATTR_256: attributes #{{[0-9]+}} = {{.*}}"min-legal-vector-width"="256"{{.*}}"prefer-vector-width"="256"

// REQUIRES: X86_ENABLED
void test_shuffle(uniform float a[], uniform float b[], uniform int n) {
    foreach(i = 0 ... n) {
        float v = a[i];
        int idx = programIndex * 2;
        float shuffled = shuffle(v, idx);
        b[i] = shuffled;
    }
}

void test_reduce(uniform float a[], uniform float* uniform result) {
    float sum = 0.0f;
    foreach(i = 0 ... 16) {
        sum += a[i];
    }
    *result = reduce_add(sum);
}

void test_math(uniform float a[], uniform float b[], uniform int n) {
    foreach(i = 0 ... n) {
        b[i] = sqrt(a[i]) + rsqrt(a[i] + 1.0f) + rcp(a[i] + 2.0f);
    }
}

void test_minmax(uniform float a[], uniform float b[], uniform int n) {
    foreach(i = 0 ... n) {
        b[i] = min(a[i], 100) + max(a[i], 10);
    }
}
