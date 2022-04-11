// This test is based on yarpgen test and checks that compiler does not crash during gather coalescing optimization.
/*
yarpgen version 2.0 (build c611daf on 2021:04:13)
Seed: 3330730858
Invocation: ../../build/yarpgen --std=ispc
*/

// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 --emit-spirv -o %t.spv
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-spirv -o %t.spv
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 --emit-spirv --opt=disable-xe-gather-coalescing -o %t.spv
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-spirv --opt=disable-xe-gather-coalescing -o %t.spv
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 --emit-spirv --opt=build-llvm-loads-on-xe-gather-coalescing -o %t.spv
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-spirv --opt=build-llvm-loads-on-xe-gather-coalescing -o %t.spv

// REQUIRES: XE_ENABLED

struct Parameters {
    uniform unsigned int64 var_0;
    uniform int16 var_2;
    uniform int32 var_10;
    uniform int32 arr_538[24];
};

task void test(void *uniform _p) {

    Parameters *uniform p = (Parameters * uniform) _p;
    {
        for (uniform int32 i_118 = ((((/* implicit */ uniform int32)p->var_2)) - (13033L)) /*0*/;
             i_118 < ((p->var_10) + (1959429349L)) /*24*/;
             i_118 +=
             ((((/* implicit */ uniform int32)((uniform unsigned int16)(uniform unsigned int8)99L))) - (95L)) /*4*/) {
            if (any((((/* implicit */ varying bool)((/* implicit */ uniform bool)p->arr_538[i_118]))))) {
                for (uniform int64 i_142 = ((((/* implicit */ uniform int64)(uniform int8)-10L)) + (10LL)) /*0*/;
                     i_142 < ((((/* implicit */ uniform int64)((uniform unsigned int8)2ULL))) + (21LL)) /*23*/;
                     i_142 += ((((/* implicit */ uniform int64)any((((/* implicit */ varying bool)none(
                                   (((/* implicit */ varying bool)((/* implicit */ uniform bool)p->var_0))))))))) +
                               (2LL)) /*2*/) {
                    // do nothing
                }
            }
        }
    }
}