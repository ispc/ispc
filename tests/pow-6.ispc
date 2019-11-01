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
    float a = float4((7.746754169464111),(3.545338869094849),(7.189665794372559),(3.906119108200073));
    float b = float4((-2.468974351882935),(-3.993634939193726),(1.319090127944946),(6.375326156616211));
    float ref = float4((0.006379491649568),(0.006380689330399),(13.491989135742188),(5923.391601562500000));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
