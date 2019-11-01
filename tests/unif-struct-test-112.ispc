export void f_v(uniform float RET[]) {
    uniform struct { uniform struct { float x, y; } q; int i; } a[3];
    a[1].q.x = 1;
    a[1].q.y = 2;
    RET[programIndex] = a[1].q.x + a[1].q.y;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
