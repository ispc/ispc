export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int count = 0;
    float a = aFOO[programIndex];
    // make sure a++ only evaluated once
    foreach_unique (ua in a++)
        ++count;
    RET[programIndex] = a; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex + 2;
}
