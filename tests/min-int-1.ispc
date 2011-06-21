
export uniform int width() { return programCount; }



export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int i = (int)a;
    RET[programIndex] = min((int)-20, i);
}

export void result(uniform float RET[]) { RET[programIndex] = -20; }

