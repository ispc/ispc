// rule: skip on arch=genx32
// rule: skip on arch=genx64

static uniform float array[10];
task void foo(uniform float f) { array[0] = f; }
task void foo(uniform float f, uniform int i) { array[i] = f; }
export void f_v(uniform float RET[]) { 
    launch foo(12.);
    launch foo(-1., 1);
    sync;
    RET[programIndex] = array[0] + array[1];
}


export void result(uniform float RET[]) {
    RET[programIndex] = 11.000000;
}
