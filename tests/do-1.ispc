export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = aFOO[programIndex]; 
    float r = 0.;
    do { 
        r += 1.; 
        x -= .25; 
    } while (x > 0.);
    RET[programIndex] = r;
}

export void result(uniform float RET[]) { RET[programIndex] = 4 * (programIndex+1); }
