// Testing int32 implementation of varying * varying
void tests_int32(uniform float RET[]) {
    varying int32 p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }


    // This array stores the expected dereferenced values to compare
    uniform int32 p_foo_res[programCount];
    p_foo_res[programIndex] = programIndex * programCount + programIndex  + 2;

    varying int32 * uniform au = &p_foo[0];

    // This ensures varying address is stored in av
    varying int32 * varying av = au + programIndex;
    // This will be varying value pointed by the varying address
    varying int32 avDeref = *av;
    if (avDeref != p_foo_res[programIndex])
         RET[programIndex] = 1;
}

export void f_v(uniform float RET[]) {
    RET[programIndex]  = 0;
    tests_int32(RET);
}


export void result(uniform float RET[]) { RET[programIndex] = 0; }
