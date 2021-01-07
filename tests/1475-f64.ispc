#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

// Testing double implementation of varying * varying
void tests_double(uniform float RET[]) {
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


task void f_v(uniform float RET[]) {
    RET[programIndex]  = 0;
    tests_double(RET);
}


task void result(uniform float RET[]) { RET[programIndex] = 0; }
