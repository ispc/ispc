
export uniform int width() { return programCount; }



float f(float);

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = f(a);
}

float f(float x) {
    if (x == 0)  return x;
    else         return x + f(x-1);
}

export void result(uniform float RET[]) { 
    uniform float fib[16] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136 };
    RET[programIndex] = fib[programIndex];
}

