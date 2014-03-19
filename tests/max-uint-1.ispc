
export uniform int width() { return programCount; }

export void f_fu(uniform float r[], uniform float a[], uniform float b) {
    unsigned int i = (unsigned int)a[programIndex];
    r[programIndex] = max((unsigned int)2, i);
    r[width() - 1] = max((unsigned int)10, (unsigned int)b);
}

export void result(uniform float r[]) { 
    r[programIndex] = 1+programIndex;
    r[0] = 2;
    r[programCount - 1] = 10;
}


