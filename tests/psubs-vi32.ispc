#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying int32 a_max = 0x7FFFFFFF, a_min = 0x80000000; // max and min signed int32
    varying int32 c = programIndex;
    if (programIndex % 4 == 0) {
        RET[programIndex * 2] = saturating_sub(a_max, c);
        RET[programIndex * 2 + 1] = saturating_sub(c, a_min);
    }
    else if (programIndex % 4 == 1) {
        RET[programIndex * 2] = saturating_sub(c, (varying int32)(c+3));
        RET[programIndex * 2 + 1] = saturating_sub(a_min, c);
    }
    else if (programIndex % 4 == 2) {
        RET[programIndex * 2] = saturating_sub(a_min, (varying int32)(-1*c));
        RET[programIndex * 2 + 1] = saturating_sub((varying int32)(-1*c), a_max);
    }
    else if (programIndex % 4 == 3) {
        RET[programIndex * 2] = saturating_sub(a_max, (varying int32)(-1*c));
        RET[programIndex * 2 + 1] = saturating_sub(a_max, (varying int32)(-1*c));
    }
}

task void result(uniform float RET[]) {
    varying int32 a_max = 0x7FFFFFFF, a_min = 0x80000000; // max and min signed int32
    varying int32 c = programIndex;
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
