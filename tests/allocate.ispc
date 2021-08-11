#include "../test_static.isph"

void noinline check_val_all(uniform float *uniform alloca_ptr, uniform float RET[], uniform float aFOO[]) {
    if (alloca_ptr[programIndex] != aFOO[programIndex]) {
        RET[programIndex] = -1;
    }
}

void noinline check_val_if(uniform float *uniform alloca_ptr, uniform float RET[], uniform float val) {
    if (*alloca_ptr != val) {
        RET[programIndex] = -1;
    }
}

void noinline check_val_for(uniform int *uniform alloca_ptr, uniform float RET[], uniform int val) {
    if (*alloca_ptr != val) {
        RET[programIndex] = -1;
    }
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;
    uniform float *uniform alloca_mem = (uniform float *uniform)alloca(4 * programCount);
    alloca_mem[programIndex] = aFOO[programIndex];
    check_val_all(alloca_mem, RET, aFOO);

    if (programIndex == (programCount / 2)) {
        uniform int index = (programCount / 2);
        uniform float *uniform alloca_mem_if = (uniform float *uniform)alloca(4);
        *alloca_mem_if = aFOO[index];
        check_val_if(alloca_mem_if, RET, aFOO[index]);
    }

    for (uniform int i = 0; i < 3; i++) {
        uniform int *uniform alloca_mem_for = (uniform int *uniform)alloca(sizeof(int));
        *alloca_mem_for = i;
        check_val_for(alloca_mem_for, RET, i);
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
