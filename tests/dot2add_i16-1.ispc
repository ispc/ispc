#include "../test_static.isph"
#define N 256

// Init a and b with positive values
void init(uniform int16 a[], uniform int16 b[]) {
    for (uniform int i = 0; i < N; i++) {
        a[i] = (uniform int16)i;
        b[i] = (uniform int16)i;
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

    foreach (i = 0 ... N / 2) {
        dst[i] = dot2add_i16packed(a_packed[i], b_packed[i], i);
    }
}

task void result(uniform float dst[]) {
    uniform int16 a[N];
    uniform int16 b[N];
    init(a, b);
    for (uniform int i = 0; i < N; i += 2) {
        uniform int result = 0;
        for (uniform int j = 0; j < 2; ++j) {
            uniform int32 product = (uniform int32)(a[i + j]) * (uniform int32)(b[i + j]);
            result += (uniform int32)(product);
        }
        dst[i / 2] = result + i / 2;
    }
}
