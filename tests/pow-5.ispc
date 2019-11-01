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
    float a = float4((8.776933670043945),(3.028291940689087),(7.100255966186523),(6.755406856536865));
    float b = float4((-8.812625885009766),(7.829904556274414),(-4.271719932556152),(-0.830200552940369));
    float ref = float4((0.000000004860408),(5857.770996093750000),(0.000230991427088),(0.204749509692192));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
