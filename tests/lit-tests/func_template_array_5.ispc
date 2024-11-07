// This test checks that multidimensional array can be created using template parameters.
// RUN: %{ispc} %s --emit-llvm-text --target=avx2-i32x8 --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define linkonce_odr void @func___Cuni1Cuni2Cuni3Cuni4Cuni5Cuni6
// CHECK: <8 x float> {{<float 4\.000000e\+00|splat \(float 4\.000000e\+00\)}}
// REQUIRES: X86_ENABLED && !MACOS_HOST

template<int N1, int N2, int N3, int N4, int N5, int N6>
noinline void func(uniform float RET[]) {
    float a[][N2][] = { { { N1 }, { N2 } }, { { N3 }, { N4 } }, { { N5 }, { N6 } } };
    RET[programIndex] = a[N1][N1][0];
}

void test(uniform float RET[]) {
    func<1, 2, 3, 4, 5, 6>(RET);
}
