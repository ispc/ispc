#include "test_static.isph"
#define N 256

// Init a and b with positive numbers
void init(uniform uint16 a[], uniform uint16 b[]) {
    for (uniform int i = 0; i < N; i++) {
        a[i] = (uniform uint16)(i % 32768);
        b[i] = (uniform uint16)(i % 32768);
    }
}
task void f_v(uniform float dst[]) {
    uniform uint16 a[N];
    uniform uint16 b[N];
    init(a, b);
    uniform uint a_packed[N / 2];
    pack2toint<uniform uint16>(a, a_packed, N);
    uniform uint b_packed[N / 2];
    pack2toint<uniform uint16>(b, b_packed, N);

    foreach (i = 0 ... N / 2) {
        dst[i] = dot2add_u16u16packed_sat(a_packed[i], b_packed[i], UINT32_MAX);
    }
}

task void result(uniform float dst[]) {
    uniform uint16 a[N];
    uniform uint16 b[N];
    init(a, b);
    for (uniform int i = 0; i < N; i += 2) {
        uniform uint result = 0;
        for (uniform int j = 0; j < 2; ++j) {
            uniform uint32 product = (uniform uint32)(a[i + j]) * (uniform uint32)(b[i + j]);
            result += product;
        }
        dst[i / 2] = saturating_add(result, UINT32_MAX);
    }
}
