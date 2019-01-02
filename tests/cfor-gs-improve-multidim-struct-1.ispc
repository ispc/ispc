
export uniform int width() { return programCount; }



struct Foo {
    uniform float udx[32][32];
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
#ifndef __NVPTX__ 
    uniform Foo f[5];
#else     /* too much shared memory allocated, nvcc fails to link */
    uniform Foo * uniform f = uniform new uniform Foo[5];
#define _UNMALLOC
#endif
    #pragma ignore warning
    cfor (uniform int i = 0; i < 5; ++i)
        #pragma ignore warning
        cfor (uniform int j = 0; j < 32; ++j)
            #pragma ignore warning
            cfor (uniform int k = 0; k < 32; ++k)
                f[i].udx[j][k] = 1000*i+100*j+k;

    int x = 1;
    RET[programIndex] = f[x+1].udx[b-4][programIndex];
#ifdef _UNMALLOC
    delete f;
#endif
}

export void result(uniform float RET[]) { RET[programIndex] = 2100 +programIndex; }
