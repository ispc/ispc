
struct Foo {
    float x;
    float a[0];
};

export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int nFloats = 3+programCount;
    varying Foo * uniform ptr = (varying Foo * uniform)(uniform new varying int32[nFloats]);
    memset(ptr, 0, nFloats*sizeof(int32));
    
    for (uniform int i = 0; i < nFloats-1; ++i)
        ptr->a[i] = i;
    ptr->x = aFOO[programIndex]; 

    #pragma ignore warning(perf)
    RET[programIndex] = ptr->a[1+programIndex]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
