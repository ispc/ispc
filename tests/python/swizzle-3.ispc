task void f_v(uniform float RET[]) {
    float<3> a = {1, 10, 100};
    float<6> b = a.zxyyxz;

    RET[programIndex] = b.x + 2 * b.y + 3 * b.z - 3 * b[3] - 2 * b[4] - b[5];
}

task void swizzle3_result(uniform float RET[]) { RET[programIndex] = 0; }

export void swizzle3_entry_point(uniform float res[]) { launch[1] f_v(res); }
export void swizzle3_result_entry_point(uniform float res[]) { launch[1] swizzle3_result(res); }
