export void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3]; 

    uniform float expected[4] = { 0x1.5bf0a8p+1, 0x1.d8e64cp+2, 0x1.415e5cp+4, 0x1.b4c904p+5 };
    float v = exp(a);

    RET[programIndex] = 0;
    #pragma ignore warning(perf)
    if (abs((v - expected[programIndex & 0x3]) / expected[programIndex & 0x3]) > 1e-5)
        RET[programIndex] = 1;
}
 
 export void result(uniform float RET[]) {
     RET[programIndex] = 0;
}
