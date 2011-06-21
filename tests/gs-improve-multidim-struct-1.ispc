
export uniform int width() { return programCount; }



struct Foo {
    uniform float udx[25][25];
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform Foo f[5];
    for (uniform int i = 0; i < 5; ++i)
        for (uniform int j = 0; j < 25; ++j)
            for (uniform int k = 0; k < 25; ++k)
                f[i].udx[j][k] = 1000*i+100*j+k;

    int x = 1;
    RET[programIndex] = f[x+1].udx[b-4][programIndex];
}

export void result(uniform float RET[]) { RET[programIndex] = 2100 +programIndex; }
