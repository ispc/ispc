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
    float v = float4((9.424787984619140),(4.000010000000000),(10.000010000000000),(10.000010000000000));
    float ref = float4((2.243343114852905),(1.386296868324280),(2.302586078643799),(2.302586078643799));
    RET[programIndex] = ok(log(v), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
