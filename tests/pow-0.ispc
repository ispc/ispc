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
    float a = float4((3.964647769927979),(4.465834140777588),(0.155828490853310),(3.837158679962158));
    float b = float4((6.809707164764404),(-3.626144647598267),(1.681804418563843),(0.000000000000000));
    float ref = float4((11846.722656250000000),(0.004399053286761),(0.043872252106667),(1.000000000000000));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
