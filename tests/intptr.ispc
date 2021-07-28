#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    RET[programIndex] = sizeof(uniform intptr_t);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 
#if (ISPC_POINTER_SIZE==32)
        4
#elif (ISPC_POINTER_SIZE==64)
        8
#else
#error Unknown pointer size
#endif
        ;
}
