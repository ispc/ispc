#include "../test_static.isph"
#define N 256

uniform int int_min = 0X80000000; // use INT_MIN to check that result is saturated

// Init a with positive and b with negative values
void init(uniform int16 a[], uniform int16 b[]) {
    for (uniform int i = 0; i < N; i++) {
        a[i] = (uniform int16)i;
        b[i] = (uniform int16)-i;
    }
}
task void f_v(uniform float dst[]) {
    uniform int16 a[N];
    uniform int16 b[N];
    init(a, b);
    uniform uint a_packed[N / 2];
    pack2toint<uniform int16>(a, a_packed, N);
    uniform uint b_packed[N / 2];
    pack2toint<uniform int16>(b, b_packed, N);
    int acc = 0X80000000 + 1;
    foreach (i = 0 ... N / 2) {
        dst[i] = dot2add_i16packed_sat(a_packed[i], b_packed[i], acc);
    }
}

task void result(uniform float dst[]) {
    uniform int16 a[N];
    uniform int16 b[N];
    init(a, b);
    uniform int acc = 0X80000000 + 1;
    for (uniform int i = 0; i < N; i += 2) {
        uniform int result = 0;
        for (uniform int j = 0; j < 2; ++j) {
            uniform int32 product = (uniform int32)(a[i + j]) * (uniform int32)(b[i + j]);
            result += (uniform int32)(product);
        }
        dst[i / 2] = saturating_add(result, acc);
    }
}
