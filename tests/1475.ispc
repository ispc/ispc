export uniform int width() { return programCount; }

// Testing int8 implementation of varying * varying
noinline int8 foo_var(varying int8 * varying av_foo) {
    return *av_foo;
}

noinline int8 foo_uni(varying int8 * uniform a_foo) {
    return *a_foo;
}

export void tests_int8(uniform float RET[]) {

    varying int8 p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }
    varying int8 * uniform a = &p_foo[0];
    varying int8 * varying av = &p_foo[0];
    varying int8 aDeref = foo_uni(a);
    varying int8 avDeref = foo_var(av);
    RET[programIndex]  = 0;
    if (avDeref != aDeref)
         RET[programIndex] = 1;
}

// Testing int16 implementation of varying * varying
noinline int16 foo_var(varying int16 * varying av_foo) {
    return *av_foo;
}

noinline int16 foo_uni(varying int16 * uniform a_foo) {
    return *a_foo;
}

export void tests_int16(uniform float RET[]) {

    varying int16 p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }
    varying int16 * uniform a = &p_foo[0];
    varying int16 * varying av = &p_foo[0];
    varying int16 aDeref = foo_uni(a);
    varying int16 avDeref = foo_var(av);
    RET[programIndex]  = 0;
    if (avDeref != aDeref)
         RET[programIndex] = 1;
}

// Testing int32 implementation of varying * varying
noinline int32 foo_var(varying int32 * varying av_foo) {
    return *av_foo;
}

noinline int32 foo_uni(varying int32 * uniform a_foo) {
    return *a_foo;
}

export void tests_int32(uniform float RET[]) {

    varying int32 p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }
    varying int32 * uniform a = &p_foo[0];
    varying int32 * varying av = &p_foo[0];
    varying int32 aDeref = foo_uni(a);
    varying int32 avDeref = foo_var(av);
    RET[programIndex]  = 0;
    if (avDeref != aDeref)
         RET[programIndex] = 1;
}

// Testing int64 implementation of varying * varying
noinline int64 foo_var(varying int64 * varying av_foo) {
    return *av_foo;
}

noinline int64 foo_uni(varying int64 * uniform a_foo) {
    return *a_foo;
}

export void tests_int64(uniform float RET[]) {

    varying int64 p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }
    varying int64 * uniform a = &p_foo[0];
    varying int64 * varying av = &p_foo[0];
    varying int64 aDeref = foo_uni(a);
    varying int64 avDeref = foo_var(av);
    RET[programIndex]  = 0;
    if (avDeref != aDeref)
         RET[programIndex] = 1;
}

// Testing float implementation of varying * varying
noinline float foo_var(varying float * varying av_foo) {
    return *av_foo;
}

noinline float foo_uni(varying float * uniform a_foo) {
    return *a_foo;
}

export void tests_float(uniform float RET[]) {

    varying float p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }
    varying float * uniform a = &p_foo[0];
    varying float * varying av = &p_foo[0];
    varying float aDeref = foo_uni(a);
    varying float avDeref = foo_var(av);
    RET[programIndex]  = 0;
    if (avDeref != aDeref)
         RET[programIndex] = 1;
}

// Testing double implementation of varying * varying
noinline double foo_var(varying double * varying av_foo) {
    return *av_foo;
}

noinline double foo_uni(varying double * uniform a_foo) {
    return *a_foo;
}

export void tests_double(uniform float RET[]) {

    varying double p_foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        p_foo[i] = i * programCount + programIndex  + 2;
    }
    varying double * uniform a = &p_foo[0];
    varying double * varying av = &p_foo[0];
    varying double aDeref = foo_uni(a);
    varying double avDeref = foo_var(av);
    RET[programIndex]  = 0;
    if (avDeref != aDeref)
         RET[programIndex] = 1;
}


export void f_v(uniform float RET[]) {

    tests_int8(RET);
    tests_int16(RET);
    tests_int32(RET);
    tests_int64(RET);
    tests_float(RET);
    tests_double(RET);

}


export void result(uniform float RET[]) { RET[programIndex] = 0; }
