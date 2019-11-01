export void f_v(uniform float RET[]) {
    varying struct { float x, y; int i; } a;
    a.x = 1;
    a.y = 2;
    RET[programIndex] = a.x + a.y;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
