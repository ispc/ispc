export uniform int width() { return programCount; }

// Testing int8 implementation of varying * varying
export void tests_int8(uniform float RET[]) {
    varying int8 p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }

    // This array stores the expected dereferenced values to compare
    uniform int8 p_foo_res[programCount];
    p_foo_res[programIndex] = programIndex * programCount + programIndex  + 2;

    varying int8 * uniform au = &p_foo[0];

    // This ensures varying address is stored in av
    varying int8 * varying av = au + programIndex;
    // This will be varying value pointed by the varying address
    varying int8 avDeref = *av;
    if (avDeref != p_foo_res[programIndex])
         RET[programIndex] = 1;
}

// Testing int16 implementation of varying * varying
export void tests_int16(uniform float RET[]) {
    varying int16 p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }

    //This array stores the expected dereferenced values to compare
    uniform int16 p_foo_res[programCount];
    p_foo_res[programIndex] = programIndex * programCount + programIndex  + 2;

    varying int16 * uniform au = &p_foo[0];

    // This ensures varying address is stored in av
    varying int16 * varying av = au + programIndex;
    // This will be varying value pointed by the varying address
    varying int16 avDeref = *av;
    if (avDeref != p_foo_res[programIndex])
         RET[programIndex] = 1;
}

// Testing int32 implementation of varying * varying
export void tests_int32(uniform float RET[]) {
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

// Testing int64 implementation of varying * varying
export void tests_int64(uniform float RET[]) {
    varying int64 p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }

    // This array stores the expected dereferenced values to compare
    uniform int64 p_foo_res[programCount];
    p_foo_res[programIndex] = programIndex * programCount + programIndex  + 2;

    varying int64 * uniform au = &p_foo[0];

    // This ensures varying address is stored in av
    varying int64 * varying av = au + programIndex;
    // This will be varying value pointed by the varying address
    varying int64 avDeref = *av;
    if (avDeref != p_foo_res[programIndex])
         RET[programIndex] = 1;
}

// Testing float implementation of varying * varying
export void tests_float(uniform float RET[]) {
    varying float p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }

    // This array stores the expected dereferenced values to compare
    uniform float p_foo_res[programCount];
    p_foo_res[programIndex] = programIndex * programCount + programIndex  + 2;

    varying float * uniform au = &p_foo[0];

    // This ensures varying address is stored in av
    varying float * varying av = au + programIndex;
    // This will be varying value pointed by the varying address
    varying float avDeref = *av;
    if (avDeref != p_foo_res[programIndex])
         RET[programIndex] = 1;
}

// Testing double implementation of varying * varying
export void tests_double(uniform float RET[]) {
    varying double p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }

    // This array stores the expected dereferenced values to compare
    uniform double p_foo_res[programCount];
    p_foo_res[programIndex] = programIndex * programCount + programIndex  + 2;

    varying double * uniform au = &p_foo[0];

    // This ensures varying address is stored in av
    varying double * varying av = au + programIndex;
    // This will be varying value pointed by the varying address
    varying double avDeref = *av;
    if (avDeref != p_foo_res[programIndex])
         RET[programIndex] = 1;
}


export void f_v(uniform float RET[]) {

    RET[programIndex]  = 0;
    tests_int8(RET);
    tests_int16(RET);
    tests_int32(RET);
    tests_int64(RET);
    tests_float(RET);
    tests_double(RET);

}


export void result(uniform float RET[]) { RET[programIndex] = 0; }
