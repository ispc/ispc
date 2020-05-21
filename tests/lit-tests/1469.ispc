// RUN: %{ispc} --target=sse2 -h %t.h %s
// RUN: %{ispc} --target=sse4 -h %t.h %s
// RUN: %{ispc} --target=avx1 -h %t.h %s
// RUN: %{ispc} --target=avx2 -h %t.h %s
// RUN: %{ispc} --target=avx512knl-i32x16 -h %t.h %s
// RUN: %{ispc} --target=avx512skx-i32x16 -h %t.h %s

// REQUIRES: X86_ENABLED

typedef float<2>    float2;
typedef bool<2>     bool2;

export void test() {
    float2 a = {1.0f, 4.0f};
    float2 b = {2.0f, 3.0f};
    bool2 cond2 = min(a.x, b.x) < min(a.y, b.y);
}

