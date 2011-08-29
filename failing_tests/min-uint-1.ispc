
export uniform int width() { return programCount; }

export void f_f(uniform float result[], uniform float aa[]) {
    unsigned int i = (unsigned int)aa[programIndex];
    result[programIndex] = min((unsigned int)2, i);
}

export void result(uniform float r[]) { 
    r[programIndex] = 2;
    r[0] = 1;
}


