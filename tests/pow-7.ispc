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

export void f_v(uniform float RET[]) {
    float a = float4((1.804677724838257),(5.206657886505127),(8.825845718383789),(6.882568359375000));
    float b = float4((8.867918014526367),(-8.687125205993652),(5.227282524108887),(5.230966091156006));
    float ref = float4((187.815383911132812),(0.000000595884217),(87848.742187500000000),(24112.632812500000000));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
