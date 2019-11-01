export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int v = aFOO[programIndex];
    // Shift only if v is greater than 2.
    if(v > 2) v = shift(v, -1);
    // Apply shift to all values of v.
    v = shift(v, 2);
    RET[programIndex] = v;
}

export void result(uniform float RET[]) {
     varying int val = programIndex + 1;
     val = val + 2;
     if(val > 2) val--;
     if (val >= programCount) val = 0;
     RET[programIndex] = val;
}
