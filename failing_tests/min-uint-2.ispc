
export uniform int width() { return programCount; }

export void f_f(uniform float r[], uniform float a[]) {
    unsigned int i = (unsigned int)a[programIndex];
    r[programIndex] =  min((unsigned int)20, i);
}

export void result(uniform float r[]) { 
    r[programIndex] = 1+programIndex;
}


