export void f_v(uniform float RET[]) { 
    RET[programIndex] = 0;
    if (true) return; 
    else RET[programIndex] = 1234.; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.000000;
}
