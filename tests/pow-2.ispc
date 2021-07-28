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
    float a = float4((4.049826145172119),(6.628306388854980),(4.623792648315430),(2.656122446060181));
    float b = float4((7.147558689117432),(0.000000000000000),(0.651920497417450),(9.655045509338379));
    float ref = float4((21962.773437500000000),(1.000000000000000),(2.713476419448853),(12477.703125000000000));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
