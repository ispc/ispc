export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i;
    if (a < 3) {
        ++a;
    }
    else {
        for (i = 0; i < b; ++i) {
            if (i == 2) break;
            ++a;
        }
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    RET[0] = RET[4] = RET[8] = RET[12] =\
        RET[16] = RET[20] = RET[24] = RET[28] =\
        RET[32] = RET[36] = RET[40] = RET[44] =\
        RET[48] = RET[52] = RET[56] = RET[60] = 2;
    RET[1] = RET[5] = RET[9] = RET[13] =\
        RET[17] = RET[21] = RET[25] = RET[29] =\
        RET[33] = RET[37] = RET[41] = RET[45] =\
        RET[49] = RET[53] = RET[57] = RET[61] = 3;
    RET[2] = RET[6] = RET[10] = RET[14] =\
        RET[18] = RET[22] = RET[26] = RET[30] =\
        RET[34] = RET[38] = RET[42] = RET[46] =\
        RET[50] = RET[54] = RET[58] = RET[62] = 5;
    RET[3] = RET[7] = RET[11] = RET[15] =\
        RET[19] = RET[23] = RET[27] = RET[31] =\
        RET[35] = RET[39] = RET[43] = RET[47] =\
        RET[51] = RET[55] = RET[59] = RET[63] = 6;
}
