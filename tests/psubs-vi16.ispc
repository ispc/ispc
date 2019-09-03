export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying int16 a_max = 0x7FFF, a_min = 0x8000; // max and min signed int16
    varying int16 c = programIndex;
    if (programIndex % 4 == 0) {
        RET[programIndex * 2] = saturating_sub(a_max, c);
        RET[programIndex * 2 + 1] = saturating_sub(c, a_min);
    }
    else if (programIndex % 4 == 1) {
        RET[programIndex * 2] = saturating_sub(c, (varying int16)(c+3));
        RET[programIndex * 2 + 1] = saturating_sub(a_min, c);
    }
    else if (programIndex % 4 == 2) {
        RET[programIndex * 2] = saturating_sub(a_min, (varying int16)(-1*c));
        RET[programIndex * 2 + 1] = saturating_sub((varying int16)(-1*c), a_max);
    }
    else if (programIndex % 4 == 3) {
        RET[programIndex * 2] = saturating_sub(a_max, (varying int16)(-1*c));
        RET[programIndex * 2 + 1] = saturating_sub(a_max, (varying int16)(-1*c));
    }
}

export void result(uniform float RET[]) {
    varying int16 a_max = 0x7FFF, a_min = 0x8000; // max and min signed int16
    varying int16 c = programIndex;
    if (programIndex % 4 == 0) {
        RET[programIndex * 2] = a_max - c;
        RET[programIndex * 2 + 1] = a_max;
    }
    else if (programIndex % 4 == 1) {
        RET[programIndex * 2] = -3;
        RET[programIndex * 2 + 1] = a_min;
    }
    else if (programIndex % 4 == 2) {
        RET[programIndex * 2] = a_min + c;
        RET[programIndex * 2 + 1] = a_min;
    }
    else if (programIndex % 4 == 3) {
        RET[programIndex * 2] = a_max;
        RET[programIndex * 2 + 1] = a_max;
    }
}
