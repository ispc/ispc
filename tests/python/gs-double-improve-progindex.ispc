//  rule: skip on cpu=tgllp
//  rule: skip on cpu=dg2

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform double udx[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        udx[i] = i + 1;
    int x = programIndex;
    RET[programIndex] = udx[x];
}

task void gb_double_improve_progindex_result(uniform float RET[]) { RET[programIndex] = 1 + programIndex; }

export void gb_double_improve_progindex_entry_point(uniform float res[], uniform float vfloat[], uniform float b) {
    launch[1] f_fu(res, vfloat, b);
}
export void gb_double_improve_progindex_result_entry_point(uniform float res[]) {
    launch[1] gb_double_improve_progindex_result(res);
}
