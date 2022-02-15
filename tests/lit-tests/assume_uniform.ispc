//; RUN: %{ispc} --emit-asm --target=avx2 %s -o - | FileCheck %s -check-prefix=CHECK_AVX2
//; RUN: %{ispc} --emit-asm --target=sse2 %s -o - | FileCheck %s -check-prefix=CHECK_SSE2
//; RUN: %{ispc} --emit-asm --target=avx512skx-x16 %s -o - | FileCheck %s -check-prefix=CHECK_AVX512
// REQUIRES: X86_ENABLED

//; CHECK_SSE2: foo1
//; CHECK_SSE2-NOT: cmp

//; CHECK_AVX2: foo1
//; CHECK_AVX2-NOT: cmp

//; CHECK_AVX512: foo1
//; CHECK_AVX512-NOT: cmp

// Case 1 : assume() can be used to choose branch/remove
inline uniform int bar1(uniform int a, uniform int b){
    if (a < b)
        return 2;
    return 5;
}
uniform int foo1(uniform int a, uniform int b){
    assume(a < b);
    return bar1(a, b);
}

//; CHECK_SSE2: foo2
//; CHECK_SSE2-NOT: test

//; CHECK_AVX2: foo2
//; CHECK_AVX2-NOT: test

//; CHECK_AVX512: foo2
//; CHECK_AVX512-NOT: test

// Case 2 : pointer null check removed.
inline void bar2(uniform int * uniform a) {
    if (a != NULL) {
        a[2] = 9;
    }
}
void foo2(uniform int a[]) {
    assume(a != NULL);
    bar2(a);
}


//; CHECK_SSE2: foo3
//; CHECK_SSE2: cmp
//; CHECK_SSE2-NOT: cmp

//; CHECK_AVX2: foo3
//; CHECK_AVX2: cmp
//; CHECK_AVX2-NOT: cmp

//; CHECK_AVX512: foo3
//; CHECK_AVX512: cmp
//; CHECK_AVX512-NOT: cmp

// Case 3 : loop : remove remainder loop.
int foo3(uniform int a[], uniform int count) {
    int ret = 0;
    assume(count % programCount == 0);
    foreach (i = 0 ... count) {
        ret += a[i];
    }
    return ret;
}


//; CHECK_SSE2: foo4
//; CHECK_SSE2-NOT: movup
//; CHECK_SSE2: movap
//; CHECK_SSE2: ret

//; CHECK_AVX2: foo4
//; CHECK_AVX2-NOT: movup
//; CHECK_AVX2: movap
//; CHECK_AVX2: ret

//; CHECK_AVX512: foo4
//; CHECK_AVX512-NOT: movup
//; CHECK_AVX512: movap
//; CHECK_AVX512: ret

// Case 4 : Create aligned mov.
typedef float<TARGET_WIDTH> AlignedFloat;
unmasked void foo4(uniform float Result[], const uniform float Source1[], const uniform unsigned int Iterations)
{
    assume(((uniform uint64)((void*)Source1) & (32 * TARGET_WIDTH)-1) == 0);
    assume(((uniform uint64)((void*)Result) & (32 * TARGET_WIDTH)-1) == 0);
    uniform AlignedFloat S1;
    S1[programIndex] = Source1[programIndex];
    const uniform AlignedFloat R = S1;
    Result[programIndex] = R[programIndex];
}
