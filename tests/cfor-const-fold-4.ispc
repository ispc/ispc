#define N(x)  (3*x+1)
// p a power of 2
#define ROUND_UP(x, p) ((x + (p-1)) & ~(p-1))

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex&0x3];
    uniform float x[ROUND_UP(N(3), 16)];
    uniform int i;
    #pragma ignore warning
    cfor (i = 0; i < ROUND_UP(N(3), 16); ++i)
        x[i] = i * i;
    RET[programIndex] = x[4];
}

export void result(uniform float RET[]) { RET[programIndex] = 16.; }
