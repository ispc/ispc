
export uniform int width() { return programCount; }

export void f_f(uniform float r[], uniform float a[]) {
    unsigned int i = (unsigned int)a[programIndex];
    r[programIndex] = max((unsigned int)2, i);
}

export void result(uniform float r[]) { 
    r[programIndex] = 1+programIndex;
    r[0] = 2;
}


