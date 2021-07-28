#include "../test_static.isph"
// Testing int64 implementation of varying * varying
void tests_int64(uniform float RET[]) {
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

task void f_v(uniform float RET[]) {
    RET[programIndex]  = 0;
    tests_int64(RET);
}


task void result(uniform float RET[]) { RET[programIndex] = 0; }
