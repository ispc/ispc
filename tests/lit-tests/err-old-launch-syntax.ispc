// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "launch" expressions no longer take '<' '>' around function call expression. 

static uniform float array[10000];

task void x(float f) {
    uniform int j;
    uniform int i = taskIndex;
    array[i] = i / 10000.;
    cfor (j = 0; j < 10000; ++j)
        array[i] = sin(array[i]);
    if (array[i] < .02)
        array[i] = i;
}
export void f_f(uniform float RET[], uniform float fFOO[]) { 
    float f = fFOO[programIndex];
    launch[10000] < x(f) >;
    sync;
    RET[programIndex] = array[9999];
}


export void result(uniform float RET[]) {
    RET[programIndex] = 9999.000000;
}
