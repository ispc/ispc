export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float val[programCount+3];
    for (uniform int i = 0; i < programCount+3; ++i)
        val[i] = 0;

    // make sure we reset the func mask in the foreach loop...
    if ((int)aFOO[programIndex] & 1)
        foreach (i = 0 ... programCount+3) {
            int ic = min(i, programCount-1);
            #pragma ignore warning(perf)
            val[i] += aFOO[ic] - 1 + i-ic;
        }

    RET[programIndex] = val[3+programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex+3;
}
