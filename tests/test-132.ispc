export void f_v(uniform float RET[]) { 
    RET[programIndex] = 0;
    #pragma ignore warning
    cif (true) return; 
    else RET[programIndex] = 1234.; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.000000;
}
