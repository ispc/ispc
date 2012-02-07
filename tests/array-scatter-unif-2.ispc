
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[100];
    // HACK to avoid @llvm.memset...
    for (uniform int i = 0; i < b*20; ++i)
        x[i] = 0;
    
    x[2*(a-1)] = b;
    RET[programIndex] = x[4]+x[5];
}

    
export void result(uniform float RET[]) { RET[programIndex] = programCount == 1 ? 0 : 5; }
