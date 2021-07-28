#include "../test_static.isph"
static float float4(uniform float a, uniform float b, uniform float c, 
                    uniform float d) {
    float ret = 0;
    for (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}



bool ok(float x, float ref) { return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-5; }

task void f_v(uniform float RET[]) {
    float a = float4((9.355601310729980),(2.201354980468750),(3.210669755935669),(0.018172996118665));
    float b = float4((6.188633918762207),(1.527074813842773),(-4.773533344268799),(-9.104317665100098));
    float ref = float4((1022368.125000000000000),(3.336671113967896),(0.003817234421149),(7026414561787904.000000000000000));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
