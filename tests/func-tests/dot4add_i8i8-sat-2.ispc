#include "test_static.isph"
#define N 256

// Init a and b with positive values
void init(uniform int8 a[], uniform int8 b[]) {
    for (uniform int i = 0; i < N; i++) {
        a[i] = (uniform int8)(i % 128);
        b[i] = (uniform int8)(i % 128);
    }
}
task void f_v(uniform float dst[]) {
    uniform int8 a[N];
    uniform int8 b[N];
    init(a, b);
    uniform uint a_packed[N / 4];
    pack4toint<uniform int8>(a, a_packed, N);
    uniform uint b_packed[N / 4];
    pack4toint<uniform int8>(b, b_packed, N);

    foreach (i = 0 ... N/4) {
        dst[i] = dot4add_i8i8packed_sat(a_packed[i], b_packed[i], INT32_MAX);
    }
}

task void result(uniform float dst[]) {
    uniform int8 a[N];
    uniform int8 b[N];
    init(a, b);
    for (uniform int i = 0; i < N; i += 4) {
        uniform int result = 0;
        for (uniform int j = 0; j < 4; ++j) {
            uniform int16 product = (uniform int16)(a[i + j]) * (uniform int16)(b[i + j]);
            result += (uniform int32)(product);
        }
        dst[i / 4] = saturating_add(result, INT32_MAX);
    }
}